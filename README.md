# Fake News Detection

The project implements a binary classifier to distinguish fake from real news articles with over 99% accuracy using Logistic Regression and Decision Tree models.

## Features
- Text preprocessing: lowercase conversion, URL/punctuation removal, tokenization.
- Feature extraction with TF-IDF vectorization.
- Model training and evaluation with confusion matrices, precision, recall, F1-scores.
- Hyperparameter tuning via GridSearchCV.
- High performance: Decision Tree at 99.6% accuracy on test set.

## Dataset
Approximately 45,000 balanced articles (fake/real).

## Tech Stack
- Python 3.x
- Pandas (data manipulation)
- Scikit-learn (ML models, TF-IDF, metrics)
- NLTK/Regex (text processing)
- Matplotlib/Seaborn (visualization)
- Jupyter Notebook for experimentation.

## Installation
1. Clone the repo:
   ```
   git clone https://github.com/Sonil15/FakeNewsDetection.git
   cd FakeNewsDetection
   ```
2. Create virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Place `fake.csv` and `true.csv` in the root.

**requirements.txt content:**
```
pandas
scikit-learn
nltk
matplotlib
seaborn
numpy
```

## Usage
1. Open `FakeNewsDetection.ipynb'
2. Run all cells to preprocess data, train models, and view results (confusion matrices, metrics).
3. Predict on new text:
   ```python
   # Example from notebook
   test_news = ["Example news text here"]
   prediction = model.predict(tfidf.transform(test_news))
   print("Fake" if prediction == 1 else "Real")
   ```

## Results
| Model              | Accuracy | Precision (True/Fake) | Recall (True/Fake) | F1-Score (True/Fake) |
|--------------------|----------|-----------------------|--------------------|----------------------|
| Logistic Regression | ~99%    | 0.99 / 0.99          | 0.99 / 0.99       | 0.99 / 0.99         |
| Decision Tree      | 99.6%   | High (detailed in report) | High             | High                |

## Future Work
- Test on recent datasets for generalizability.
- Advanced models: LSTM, BERT for semantic understanding.

## Contact
Sonil Negi - sonilnegi088@gmail.com

