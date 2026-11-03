"""Everything the app asks of the model: topic, confidence and why."""
import json
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MODEL_DIR
from .taxonomy import TOPIC_KEYS

MIN_WORDS = 5
UNSURE_BELOW = 0.5
EXPLAIN_STEPS = 16
MARK_SHARE = 0.25
GATE_FILE = "news_gate.json"


class NewsGate:
    """Logistic regression over BERT's pooled output: does this read like news?

    Topic confidence can't answer this. The classifier must pick one of 13 desks,
    so it is often very sure about text that isn't news at all.
    """

    def __init__(self, path):
        saved = json.loads(path.read_text())
        self.mean = torch.tensor(saved["mean"])
        self.scale = torch.tensor(saved["scale"])
        self.weights = torch.tensor(saved["weights"])
        self.intercept = saved["intercept"]
        self.threshold = saved["threshold"]

    def score(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(((pooled - self.mean) / self.scale) @ self.weights + self.intercept)


@dataclass
class Prediction:
    ranked: list[tuple[str, float]]
    word_count: int
    was_cut_short: bool
    looks_like_news: bool = True

    @property
    def topic(self) -> str:
        return self.ranked[0][0]

    @property
    def confidence(self) -> float:
        return self.ranked[0][1]

    @property
    def is_unsure(self) -> bool:
        return self.confidence < UNSURE_BELOW


class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        calibration = json.loads((model_dir / "calibration.json").read_text())
        self.temperature = calibration["temperature"]
        self.max_length = calibration["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, dtype=torch.float32
        ).eval()
        gate_path = model_dir / GATE_FILE
        self.gate = NewsGate(gate_path) if gate_path.exists() else None

    def pooled(self, encoded) -> torch.Tensor:
        return self.model.bert(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        ).pooler_output

    def _encode(self, texts, **options):
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            **options,
        )

    @torch.inference_mode()
    def classify_many(self, texts: list[str], batch_size: int = 16) -> list[Prediction]:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._encode(batch)
            pooled = self.pooled(encoded)
            probabilities = torch.softmax(self.model.classifier(pooled) / self.temperature, dim=-1)
            is_news = (
                self.gate.score(pooled) >= self.gate.threshold
                if self.gate
                else torch.ones(len(batch), dtype=torch.bool)
            )
            for text, row, mask, news in zip(batch, probabilities, encoded["attention_mask"], is_news):
                order = row.argsort(descending=True).tolist()
                results.append(
                    Prediction(
                        ranked=[(TOPIC_KEYS[i], float(row[i])) for i in order],
                        word_count=len(text.split()),
                        was_cut_short=bool(mask.sum() >= self.max_length),
                        looks_like_news=bool(news),
                    )
                )
        return results

    def classify(self, text: str) -> Prediction:
        return self.classify_many([text])[0]

    def explain(self, text: str, topic: str) -> list[dict]:
        """Integrated gradients, grouped from word pieces back to whole words.

        Each item is a character span of `text` with a score from -1 to 1:
        positive words pushed the model toward `topic`, negative ones away.
        The baseline is [MASK], which read as "a word goes here" and picked out
        far better words than the empty [PAD] baseline in a masking test.
        """
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_tensors="pt", return_offsets_mapping=True,
        )
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        special = torch.tensor(
            self.tokenizer.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True),
            dtype=torch.bool,
        )
        baseline_ids = ids.clone()
        baseline_ids[0, ~special] = self.tokenizer.mask_token_id

        embed = self.model.bert.embeddings.word_embeddings
        with torch.no_grad():
            start, end = embed(baseline_ids), embed(ids)
        alphas = (torch.arange(EXPLAIN_STEPS, dtype=torch.float32) + 0.5) / EXPLAIN_STEPS
        path = start + alphas.view(-1, 1, 1) * (end - start)
        path = path.detach().requires_grad_(True)

        logits = self.model(
            inputs_embeds=path, attention_mask=mask.repeat(EXPLAIN_STEPS, 1)
        ).logits
        target = TOPIC_KEYS.index(topic)
        margin = logits[:, target] - logits.mean(dim=1)
        gradients = torch.autograd.grad(margin.sum(), path)[0].mean(dim=0)
        tokens = ((end - start)[0] * gradients).sum(dim=-1).detach()

        words: dict[int, dict] = {}
        offsets = encoded["offset_mapping"][0].tolist()
        for index, word_id in enumerate(encoded.word_ids(0)):
            if word_id is None:
                continue
            begin, finish = offsets[index]
            word = words.setdefault(word_id, {"start": begin, "end": finish, "score": 0.0})
            word["end"] = finish
            word["score"] += float(tokens[index])

        spans = list(words.values())
        peak = max((abs(s["score"]) for s in spans), default=0.0) or 1.0
        for span in spans:
            span["score"] /= peak
        return spans


def mark_spans(spans: list[dict]) -> list[dict]:
    """Keep only the words strong enough to be worth a reader's attention."""
    return [span for span in spans if abs(span["score"]) >= MARK_SHARE]
