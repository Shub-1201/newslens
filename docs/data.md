# Data

## Why three datasets

BBC News alone covers 5 topics from one publisher in 2004 and 2005. Two more sources add topics, publishers and time.

| Source | Rows kept | Period | What it adds |
|---|---:|---|---|
| BBC News | 2,102 | 2004 to 2005 | Clean long articles |
| MN-DS | 8,752 | 2019 | Hand-labelled articles from 215 outlets, and the only source for conflict and disasters |
| HuffPost | 6,595 | 2012 to 2022 | Headlines with short descriptions, in a modern voice |

AG News was left out because its licence is unknown. 20 Newsgroups and Reuters are older and don't fit the task.

## One set of desks

Each dataset labels news differently, so `taxonomy.py` maps them to 13 shared desks: Business & Work, Politics, Sport, Entertainment & Arts, Science & Technology, Health, Crime & Law, Conflict & War, Disasters & Weather, Environment, Lifestyle & Travel, Education and Religion.

Vague labels were dropped rather than forced in. HuffPost's IMPACT and BLACK VOICES, and MN-DS's society and human interest, say more about angle or audience than topic. HuffPost's are filtered as it loads, so the 1,700 unmapped rows below are all from MN-DS.

## Cleaning

| Step | Rows |
|---|---:|
| Loaded (HuffPost capped at 600 per desk, newest first) | 19,742 |
| Dropped, no matching desk | 1,700 |
| Dropped, duplicates | 589 |
| Dropped, under 5 words | 4 |
| **Corpus** | **17,449** |

Duplicates are found two ways: on the full normalised text, and on the first 100 normalised characters, which catches syndicated copies with different endings.

A finding worth knowing: the published BBC dataset has 103 duplicates, and 45 of its test rows also appear in its training set. Anyone scoring on that split gets inflated accuracy, which is why this project makes its own.

## Splits

The corpus is split 80/10/10 into 13,959 training, 1,745 validation and 1,745 test stories, stratified on desk and source together. `assert_no_leakage` stops the run if any key or opening crosses splits.

Conflict and disasters exist only as long articles, while the app also receives headlines. So for those topics, the training set gets an extra copy of each story's headline, bringing it to 15,263 rows. This happens only after splitting, so no headline can leak into evaluation.

## Feedback

Each saved verdict records the time, predicted desk, confidence, right or wrong, the correct desk and the story text. It is meant for review and retraining. The app tells testers the text is stored.
