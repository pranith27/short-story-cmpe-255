# Deep Learning Advancements in Anomaly Detection

**Pranith Varma Pakalapati — San Jose State University, CMPE 255 Short Story**

A short story on a current survey paper in data mining and machine learning: 180+ deep-learning anomaly detection studies, organised into reconstruction-based, prediction-based, and hybrid methods — with applications, comparison tables, and open research directions.

## Links

- **Source paper:** https://arxiv.org/abs/2503.13195
- **Medium article:** https://medium.com/@pranithvarma.234/deep-learning-changed-anomaly-detection-the-hard-part-is-still-the-data-f5408c0ca2a4
- **PPT Slides :** https://drive.google.com/file/d/1rhRWB7bvyOBENjnixOmv1nHqsQEXARIc/view?usp=sharing
- **Video:** https://youtu.be/TxQmFk3l858

## About this assignment


I chose *"Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey"* by Haoqi Huang, Ping Wang, Jianhua Pei, Jiacheng Wang, Shahen Alexanian, and Dusit Niyato (arXiv:2503.13195v1, 17 March 2025, IEEE Internet of Things Journal). It is a survey, not a single new model — which is exactly why it works as a short story: it maps a whole field, so the story has a beginning, a map, and a conclusion.

## The paper at a glance

- **Scope:** 180+ studies reviewed, mostly 2019–2024, from IEEE, ACM, Springer and Elsevier journals and conferences such as AAAI, CCS and ICCV.
- **Authors:** Huang and Wang (York University), Pei (Huazhong University of Science and Technology), Wang and Niyato (Nanyang Technological University), Alexanian (York University).
- **Why this survey:** earlier surveys mostly stopped around 2020 and missed the newer models — conditional and cycle-consistent GANs, GAN + VAE combinations, diffusion models, and hybrid methods.
- **Contribution:** a clear taxonomy of methods, structured comparison tables, and a full section on hybrid approaches that combine deep learning with traditional data mining.

## What is an anomaly?

An anomaly is an observation that deviates significantly from what a system expects. It is usually a tiny fraction of the data — often under 1% — yet frequently the most important part of it: one fraudulent transaction in a million, one drifting sensor in a factory, one lesion on a scan that looks almost like healthy tissue, one cyberattack hidden in a log file of ordinary traffic.

The survey starts with the data, not the model. Text, audio, image, video and tabular data each produce different anomalies, and time-series data adds its own kinds: point anomalies (one strange value), sequence anomalies (a strange stretch), and group anomalies (a strange cluster). Naming the data type and the anomaly type before choosing a model is the first practical lesson of the paper.

## The map: three families of methods

Everything in the survey hangs off this taxonomy:

1. **Reconstruction-based methods** — learn what normal looks like, rebuild the input, and flag large reconstruction error. Autoencoders, variational autoencoders (VAEs), GANs, and diffusion models. They work unsupervised, on high-dimensional data, with no labelled anomalies at all.
2. **Prediction-based methods** — learn the sequence, predict the next value, and flag large prediction error. RNNs, LSTMs and GRUs, Transformers, CNNs, and graph neural networks. Their strength is temporal and relational modelling; their weakness is that a good forecast is not the same as a good anomaly score — the threshold decides.
3. **Hybrid methods** — deep learning supplies the features, a traditional method supplies a boundary you can explain. Clustering plus deep learning, normalising flows, and Deep SVDD. FastFlow, a normalising flow, reached 99.4% AUC on MVTec AD while staying fast at inference.

The map also tells you how to choose: reconstruction asks "does it look normal?", prediction asks "does it follow?". No time order in the data means reconstruction.

## Where it is used

- **Time-series:** traffic monitoring, power grids (the MENSA intrusion detection system), healthcare vitals monitored continuously.
- **Non-temporal:** finance (fraud detection, credit scoring, trading activity) and cybersecurity (intrusion detection).
- **Visual:** medical imaging (X-ray, brain, CT), video surveillance, and manufacturing defect detection.

## Headline findings

- FastFlow: **99.4% AUC on MVTec AD**, with high inference efficiency — the strongest single result in the survey.
- Chest CT: **AUROC > 0.90 for 18 abnormalities** — deep methods work where the signal is subtle and close to normal tissue.
- Transformer models: **F1 > 0.96** in a multimodal setting (best case reported), and up to **+17% F1** from better inference in another study.
- These numbers come from different datasets and are **not directly comparable**. Judge a model by AUC, F1, and recall on the anomaly class — never by accuracy alone on imbalanced data, where 99% accuracy can mean 0% anomaly recall.

## Ablations and trade-offs — my reading

Reading across the papers the survey reviews, four conclusions stand out:

- **The anomaly score matters more than the architecture.** Swapping an autoencoder for a VAE, or an LSTM for a Transformer, moves the score a few points. The wrong score or threshold breaks the system.
- **The training data matters more than the model.** Clean normal data, imbalance handling, and noisy labels change results more than adding another layer.
- **Accuracy trades against cost.** The best models are the heaviest. For real-time use, a lighter model answering in milliseconds can be the right choice even if its AUC is a point lower.
- **The first ablation to run:** fix the model, vary the threshold, and plot precision against recall. It tells you more about deployment than any architecture swap.

## Open issues

Data collection is still the bottleneck — anomalies are rare, labelling is costly and needs experts, and synthetic data does not fully solve it. Computational complexity, explainability (attention, LIME, SHAP), and handling diverse anomaly types (point, sequence, group) remain open. My own addition: shared benchmarks — same datasets, same splits, same metrics — until then, headline numbers will keep looking better than they are.

## My take

Hybrids are the point, not a compromise: deep learning supplies the features, a traditional method supplies a boundary you can explain. And the deployment decision is usually the threshold, the training data, and explainability — not the architecture.

## Contents

- `figures/` — all 6 diagrams used in the slides and the article (taxonomy map, autoencoder diagram, prediction diagram, class-imbalance chart, headline results chart, accuracy-vs-cost chart).
- `medium-article/` — the full Medium article (.md and .docx), about 3,900 words, with the same 6 figures.
- `research-paper/` — the survey paper: *Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey* (Huang et al., arXiv:2503.13195v1).
- `ppt/` — the 18-slide deck (.pptx and .pdf). The full speaking script is in each slide's Speaker Notes — about 3,100 words, roughly 19–20 minutes at a natural pace.

## References

- Huang, H., Wang, P., Pei, J., Wang, J., Alexanian, S., & Niyato, D. (2025). *Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey.* arXiv:2503.13195v1. https://arxiv.org/abs/2503.13195
- Key results cited here are as reported in the survey; see the survey's reference list for the full citations of the individual papers (MENSA IDS, FastFlow, Deep SVDD-VAE, and others).
