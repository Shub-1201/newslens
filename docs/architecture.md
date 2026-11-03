# Architecture

## The path of one story

1. The app tokenises the text and keeps the first 128 tokens, about 100 words.
2. BERT produces one 768-number summary of the text.
3. The news check reads that summary. Text below its threshold is set aside.
4. The classifier turns the same summary into 13 scores, and temperature scaling converts them to confidence.
5. Below 50% confidence, the app asks for an editor. A confidence under 50% means the model itself thinks it's more likely wrong than right.
6. Otherwise it names the desk, then runs integrated gradients to find the words that swayed it.
7. The tester's verdict goes to a CSV for later review.

Steps 1 to 5 use one forward pass. Step 6 adds 16 more, which is why a single story takes about 3 seconds on a CPU.

## Decisions and why

### Partial fine-tuning, with the frozen layers cached

Only the top 2 of BERT's 12 layers train, plus the pooler and classifier. On 2 CPU cores with no GPU, full fine-tuning would take many hours.

The bottom 10 layers never change, so their output for each text is computed once and stored on disk as `float16`. Each training step then runs only the top layers. A test confirms this path gives the same result as a normal forward pass, and that the `float16` storage barely moves predictions.

Batches are grouped by length so short texts don't pay for padding. This brought training to about 70 minutes.

### Calibrated confidence

Neural networks tend to be overconfident, and the app shows confidence to people. Temperature scaling learns one number on validation data that adjusts the scores without changing which desk wins. Here it's 1.10. On full stories, stated confidence and real accuracy now sit within 2.6 points.

### A separate news check

The first version relied on low confidence to catch non-news, and that failed: keyboard gibberish scored 91% Science & Technology. A classifier must choose among its 13 desks, so confidence says nothing about whether any of them apply.

Four approaches were tested at a fixed cost of 2% of real news:

| Method | Code caught | Gibberish caught | Emails caught | Tweets caught |
|---|---:|---:|---:|---:|
| Top confidence | 1% | 7% | 11% | 25% |
| Energy score | 2% | 21% | 13% | 40% |
| Mahalanobis distance | 8% | 67% | 16% | 48% |
| Logistic regression, kind held out | 99% | 100% | 66% | 87% |

The logistic regression needs examples of non-news, but it won by a wide margin. It's a 48 KB file and adds no noticeable time. The "kind held out" row is its result on text it had never seen, which is the fair comparison with the unsupervised methods.

### Explanations from a [MASK] baseline

Integrated gradients compares the real story with a neutral baseline. An empty [PAD] baseline gave poor words: on a story about drone strikes, it marked "without" and missed "shelling". A [MASK] baseline, which reads as "some word goes here", did much better.

To compare them, each method's top 5 words were masked to see how far confidence fell. Bigger is better.

| Baseline and steps | Mean confidence drop | Seconds per story |
|---|---:|---:|
| [PAD], 8 steps | 0.38 | 0.8 |
| [PAD], 24 steps | 0.40 | 2.4 |
| Word-by-word occlusion | 0.42 | 1.5 |
| [MASK], 8 steps | 0.45 | 0.9 |
| **[MASK], 16 steps (used)** | **0.48** | **1.7** |
| [MASK], 24 steps | 0.49 | 2.6 |

This test removes words using [MASK], so it leans toward that baseline. The choice also rests on reading the marked words, which were clearly better.

The display then drops common words and shows only the strongest eighth of the text, with a minimum of four.

### Why these tools

Gradio gives a working interface with file upload, tables and queueing in little code, and it runs free on Hugging Face Spaces. All user text is escaped before going into HTML, and a test covers that.
