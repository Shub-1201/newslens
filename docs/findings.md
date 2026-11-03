# Findings

All figures come from `artifacts/results/metrics.json`, measured on the saved model with data held out from training and tuning.

## Overall

| Measure | Result |
|---|---:|
| Accuracy, 1,745 stories | 82.9% |
| Macro-F1 | 0.83 |
| Validation macro-F1 by epoch | 0.79, 0.82, 0.82 |
| Training time, 2 CPU cores | 71 minutes |

The saved `float16` model reproduces the training run's accuracy.

## By source

| Source | Stories | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| BBC | 211 | 95.3% | 0.96 |
| MN-DS | 874 | 84.6% | 0.84 |
| HuffPost | 660 | 76.8% | 0.77 |

BBC's high score is mostly a reflection of the task: five easy topics and long, clean articles. HuffPost is the hardest because it has only a headline and a short description.

## By desk

| Desk | F1 |
|---|---:|
| Conflict & War | 0.93 |
| Sport | 0.93 |
| Entertainment & Arts | 0.87 |
| Environment | 0.86 |
| Disasters & Weather | 0.86 |
| Religion | 0.84 |
| Business & Work | 0.84 |
| Lifestyle & Travel | 0.83 |
| Education | 0.81 |
| Politics | 0.80 |
| Health | 0.80 |
| Crime & Law | 0.73 |
| Science & Technology | 0.69 |

The most common mistakes make sense. Science & Technology stories go to Business 13 times, and Politics stories to Science & Technology 11 times. Much modern tech news really is business or policy news, so some of these are label disagreements rather than errors.

## Confidence

| Commit above | Stories answered | Accuracy when answering |
|---|---:|---:|
| 40% | 97.9% | 84.2% |
| 50% | 93.5% | 85.6% |
| 60% | 87.7% | 87.9% |
| 70% | 81.4% | 90.0% |
| 80% | 72.1% | 92.9% |

The app uses 50%, because below that the model itself expects to be wrong. Including the news check, it gives a desk to 91.5% of stories and is right on 85.7% of them. A newsroom wanting fewer mistakes could raise the bar and send more to an editor.

These thresholds were read from the test set, so treat them as a description rather than a tuned result.

## Headlines are a different problem

Headlines alone reach 65.6% accuracy with an average confidence of 78%, so they're overconfident by 12.7 points. Calibration was fitted on full stories, so it doesn't carry over. The app warns on text under 25 words instead of hiding this. Fitting a second temperature for short text is the obvious next step.

## The news check

The threshold was set on validation data to lose 2% of real news.

| Text | Caught, kind seen in training | Caught, kind held out |
|---|---:|---:|
| Code | 99% | 99% |
| Gibberish | 100% | 100% |
| Text messages | 95% | 89% |
| Tweets | 94% | 87% |
| Emails | 93% | 66% |
| Quotes | 97% | 46% |
| Film reviews | 95% | 23% |

The right column is the honest guide to unfamiliar text. Film reviews are the hardest because they read much like entertainment journalism.

On real news, it passes 97.5% of test stories and 98.9% of headlines. The cost falls unevenly: it sets aside 9% of health stories and 6% of lifestyle stories, likely because advice-style writing resembles non-news. When that happens, the slip still shows the closest desks.

## Sample stories

Hand-written, modern-style stories scored 24 of 26, and all five non-news samples were set aside. The two misses were a crypto founder's sentencing (Business, labelled Crime & Law) and a hurricane evacuation (Environment, labelled Disasters & Weather). This is a smoke test, not a benchmark.

## What to do next

1. Collect real tester verdicts, since they're the only labels that reflect how people use the app.
2. Calibrate short text separately.
3. Add more recent labelled news. The 2004 to 2022 range is the clearest risk to real-world accuracy.
4. Add more kinds of non-news, especially reviews and opinion writing.
5. Try multi-label output, so a story about a crypto fraud trial can sit on two desks.
