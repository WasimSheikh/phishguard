# PhishGuard

A small phishing URL detector I built to practice logistic regression. It only looks at the text of the URL: length, digit count, special characters, subdomains, whether the host is an IP, https vs http, and a short list of suspicious words.

Two models get those same features. One is logistic regression written from scratch in NumPy (sigmoid, cross-entropy, gradient descent, L2). The other is sklearn's `LogisticRegression`. The decision threshold is picked on a validation set by F1, then everything is scored on a held-out test set.

This is a course project, not a production filter.

## How to run

```bash
pip install -r requirements.txt
python -m phishguard train
python -m phishguard score 'https://example.com'
```

`train` downloads the PhiUSIIL CSV from UCI if it is not already in `data/`, fits both models, and writes `models/phishguard.npz` plus the charts. Tests:

```bash
pytest -q
```

## Results

80,000 URLs drawn from PhiUSIIL (seed 7), stratified 70/15/15. Positive class is phishing. Numbers below are on the 12,000-row test set.

| model | threshold | precision | recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy LR | 0.25 | 0.955 | 0.970 | 0.962 | 0.993 |
| sklearn LR | 0.40 | 0.969 | 0.959 | 0.964 | 0.994 |

The two fits land in the same place. Sklearn's solver is a bit better calibrated, so its threshold sits closer to 0.5. The NumPy run needs a lower cut to get the same F1.

![ROC curves](charts/roc.png)

![NumPy confusion matrix](charts/confusion_numpy.png)

## Data

[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (Prasad and Chandra, *Computers & Security*, 2024). Also on [Mendeley](https://data.mendeley.com/datasets/shwpxscxy2/1). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The raw file is about 50MB and is not checked in. `python -m phishguard train` (or `python scripts/download_data.py`) pulls it from UCI and keeps `URL` plus a phishing label.

## Limitations

Lexical features only. No HTML, no DNS, no hosting or reputation signals. In this dataset every HTTP URL is phishing and every legitimate URL is HTTPS, and digit counts are also very skewed, so the numbers above look stronger than they would on the live web. Do not use this as a real phishing filter.

Built after my DeepLearning.AI / Stanford Online course [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/account/accomplishments/verify/VW0IAEFG1VKB).
