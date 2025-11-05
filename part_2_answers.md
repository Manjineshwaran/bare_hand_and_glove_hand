## Part 2: Reasoning-Based Questions (Write-up)

### Q1: Choosing the Right Approach
I’d start with detection to see if the label is present or not. It’s simple, fast, and only needs a box. If detection is noisy, I’d add a tiny classifier on the cropped label to confirm “label vs. no‑label.” If that still fails, I’d try segmentation to measure label shape/coverage. I’d also try a quick template match as a baseline.

### Q2: Debugging a Poorly Performing Model
Check for domain shift: compare a few training vs. factory images side by side. Manually review 50 labels to catch annotation errors. Look at top false negatives/positives and their confidences. Toggle augmentations and see if they help or hurt. Fine‑tune on 50–100 labeled factory images and test on a held‑out factory set.

### Q3: Accuracy vs Real Risk
Accuracy isn’t the right metric here. Focus on recall for defects (don’t miss bad items) and monitor precision to limit false rejects. Use a confusion matrix and PR curve to tune a threshold. Pick a threshold that meets a target miss rate and track false rejects in production. A recall‑weighted F‑beta is better aligned with risk.

### Q4: Annotation Edge Cases
Keep blurry/partial objects if they happen in real life, but tag them with a quality flag. If an object is truly unknowable, exclude it or mark as “ignore.” Including them improves robustness but adds label noise. Do an A/B test with and without them to pick the right policy.

