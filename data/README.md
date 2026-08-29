# Data

Raw data isn't checked into this repo (the primary review dataset alone is ~3M rows / several GB). To reproduce the pipeline, download:

- **Amazon Books Reviews** — [Kaggle: mohamedbakhet/amazon-books-reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews). The notebook uses [`kagglehub`](https://pypi.org/project/kagglehub/) to fetch this directly and draws a reproducible random sample of 10,000 rows (`random_state=1`).
- **GoEmotions** — [google-research/google-research: goemotions](https://github.com/google-research/google-research/tree/master/goemotions). Used only to train the emotion classifiers; the pre-trained DistilBERT model (`bhadresh-savani/distilbert-base-uncased-emotion`) is downloaded automatically via Hugging Face `transformers` and requires no separate training if you just want to reproduce the scoring step.
- Supporting book-metadata dataset used for the DS1+DS4 merge (see notebook for the exact source and join logic).

Once downloaded, point the notebook's data-loading cells at your local paths and run end to end.
