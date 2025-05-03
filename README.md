# Amogh Mittal

# Movie Review Prediction

This project implements a machine learning pipeline to classify Amazon product reviews based on their textual content.  
We predict the review score (1 to 5 stars) based on both the review's summary and full text fields.

---

## Data Exploration

- Loaded `train.csv` containing 1.7 million reviews.
- Observed missing values in the `Score` column (used to separate train/test).
- Noticed heavy class imbalance — many reviews are 5-star.
- The text fields (`Summary` + `Text`) vary in length from very short to very long reviews.

---

## Feature Extraction / Engineering

- Combined the `Summary` and `Text` fields into a single text feature.
- Performed basic text preprocessing:
  - Lowercasing
  - Removing non-alphanumeric characters
- Created a sparse TF-IDF representation:
  - **Max Features**: 20,000 (more than 20k was too expensive computationaly, was taking a lot of time)
  - **N-gram Range**: (1,2) (unigrams and bigrams)
  - **Sublinear TF Scaling**: Enabled for smoother term frequencies
- Handled missing text values by filling with empty strings.

---

## Model Creation and Assumptions

- Chose **Logistic Regression** as the main model:
  - Well-suited for high-dimensional sparse data (TF-IDF).
  - Efficient and scalable for large datasets.
- Assumptions:
  - Review texts contain sufficient signal to predict the score.
  - Unigrams and bigrams capture important sentiment and intent phrases.
  - TF-IDF features are enough without additional numeric or metadata features.

---

## Model Tuning

- Set the Logistic Regression hyperparameters:
  - `solver='saga'` (supports sparse matrices)
  - `max_iter=2000` (tried 1000 too earlier)
- Some other Logistic Regression hyperparameters I had set but didn't work well:
  - `C=0.5, 1` (controls regularization strength)
  - `penalty='l2'` (ridge regularization)
  - `class_weight='balanced'` (to address class imbalance of 5 star reviews)
- Stratified train-validation split (80/20) to maintain class distribution during evaluation.
- Tried performing grid search, but took lot of training time.

---

## Model Evaluation / Performance

- **Validation Accuracy**: Approximately **64%–65%** on stratified validation split.
- Full retraining was performed on all labeled data before final test prediction.
- Final `submission.csv` is created by merging predicted scores onto test IDs.
- Strange that I got better accuracy when training 900,000 rows than 1 million!

---

## Struggles / Issues / Open Questions

- **Bigram TF-IDF** led to very large feature spaces; initial runs took very long (>1 hour).
- **RAM and computation constraints** forced limiting TF-IDF `max_features` to 20,000.
- **Dataset of almost 1.7 million** datapoints was not feasible to train everytime so had to choose random rows (tested with 500k, 900k, 1 million rows).
- **RidgeClassifier** was tested but slightly underperformed compared to Logistic Regression (~1% lower accuracy).
- **Random Forest** was performed but underperformed in comparison on Logistic Regression.
- **LinearSVC** was tried but was computationally too expensive on full data.
- **Ensemble Model or Model Stacking** was tested with Logistic and Ridge Classifiers but again Logistic Regression gave the best accuracy.
- **Handling class imbalance**: `class_weight='balanced'` should've helped better but didn't as 5-star dominance remained an issue.
- **Potential future improvements**:
  - More advanced feature engineering (e.g., sentiment scores, review lengths).
  - Model stacking or ensembling on a reduced dataset.
  - Hyperparameter tuning (GridSearchCV) on a reduced dataset.

---

## Conclusion

The project successfully built a clean, efficient text classification pipeline using TF-IDF and Logistic Regression.  
Despite computational constraints, a robust and scalable model was developed with strong validation performance.

