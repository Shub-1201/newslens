"""Keep testers' verdicts, so real-world accuracy can be measured later."""
import csv
import threading
from datetime import datetime, timezone

from .config import FEEDBACK_FILE

FIELDS = ["saved_at", "predicted", "confidence", "verdict", "correct_desk", "text"]
_lock = threading.Lock()


def save(predicted, confidence, verdict, correct_desk, text, path=FEEDBACK_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predicted": predicted,
        "confidence": round(confidence, 4),
        "verdict": verdict,
        "correct_desk": correct_desk,
        "text": text,
    }
    with _lock:
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
