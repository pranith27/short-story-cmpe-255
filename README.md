# Deep Learning Advancements in Anomaly Detection — CMPE 255 Short Story

**Author:** Pranith Varma Pakalapati, San Jose State University (CMPE 255)
**Source paper:** Haoqi Huang, Ping Wang, Jianhua Pei, Jiacheng Wang, Shahen Alexanian, and Dusit Niyato, "Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey," arXiv:2503.13195v1, 17 March 2025, IEEE Internet of Things Journal.
**Paper link:** https://arxiv.org/abs/2503.13195

## What this is

A short-story assignment on a current survey paper in data mining / machine learning. The survey reviews 180+ deep-learning anomaly detection studies (2019–2024) and organises them into reconstruction-based, prediction-based, and hybrid methods, with applications, comparison tables, and open research directions.

## Contents

- `Deep_Learning_Anomaly_Detection_Short_Story.pptx` — 18-slide deck (16:9, white pages, minimal design), with the full speaking script in each slide's Speaker Notes
- `Deep_Learning_Anomaly_Detection_Short_Story.pdf` — the same deck as PDF
- `Presentation_Script_15-20min.txt` — full speaking script, about 3,100 words, roughly 19–20 minutes at a natural pace (about 17 minutes brisk), with per-slide timings and delivery cues. If running long, shorten Slides 4 and 13 first.
- `Deep_Learning_Anomaly_Detection_Article.md` / `.docx` — Medium-style article (original paraphrase, with figures and my own take)
- `Deep_Learning_Anomaly_Detection_Article_Full.md` / `.docx` — full expanded Medium article (~3,900 words) with all 6 figures, same diagrams as the PPT — copy-paste this one into Medium
- `figures/` — 6 figures used in the slides and article (taxonomy map, autoencoder diagram, prediction diagram, class-imbalance chart, headline results chart, accuracy-vs-cost chart)
- `2503.13195v1.pdf` — source survey PDF

## Links to submit

- Medium article: https://medium.com/@pranithvarma.234/deep-learning-changed-anomaly-detection-the-hard-part-is-still-the-data-f5408c0ca2a4
- SlideShare presentation: _add your SlideShare link here after uploading the deck_
- Presentation (Google Drive, public link): https://drive.google.com/file/d/1RBkYQbz-8amTU-DYWbHW1eyqYd3fdyLA/view?usp=sharing
- Video: record yourself narrating the deck using `Presentation_Script_15-20min.txt`, then upload the recording to this repo

## Headline findings (as reported in the survey)

- FastFlow (normalising flow + ResNet / Vision Transformer features): **99.4% AUC on MVTec AD**, with high inference efficiency.
- Medical imaging: **AUROC > 0.90 for 18 abnormalities** in chest CT volumes.
- Transformer-based models: **F1 > 0.96** in a multimodal setting (best case reported).
- These numbers come from different datasets and are not directly comparable — the article and script explain why AUC, F1 and recall on the anomaly class matter more than accuracy.

## My take

Hybrids are the point, not a compromise: deep learning supplies the features, a traditional method (clustering, normalising flows, Deep SVDD) supplies a boundary you can explain. And the deployment decision is usually the threshold, the training data, and explainability — not the architecture.
