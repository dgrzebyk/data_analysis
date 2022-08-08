import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import f1_score

train_set = pd.read_csv("")
X_test = pd.read_csv("")

cols_to_drop = ['ID', 'advice_to_mgmt', 'date', 'location']
text_features = ['job_title', 'summary', 'positives', 'negatives']
num_features = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6']
cat_features = ['Place', 'status']

# Rating of a company does not depend on review ID, advice_to_mgmt contains 1/3rd NaNs
train_set.drop(columns=cols_to_drop, inplace=True)
# X_test.drop(columns=cols_to_drop, inplace=True)

# Data cleaning
train_set = train_set.drop_duplicates()
train_set.dropna(inplace=True)
# X_test = X_test.drop_duplicates()
# X_test.dropna(inplace=True)

# Ensuring correct dtype
train_set[text_features] = train_set[text_features].astype("str")
X_test[text_features] = X_test[text_features].astype("str")

# Train-test split
X_train = train_set.drop(columns=['overall'])
y_train = train_set['overall']

# Defining preprocessing steps
vect = CountVectorizer()
tfidf = TfidfTransformer()
tfidf_vect = TfidfVectorizer()

ct = make_column_transformer((StandardScaler(), num_features),
                             (OneHotEncoder(handle_unknown='ignore'), cat_features),
                             # (tfidf_vect, 'job_title'), (tfidf_vect, 'summary'), (tfidf_vect, 'positives'), (tfidf_vect, 'negatives'),
                             (vect, 'job_title'), (vect, 'summary'), (vect, 'positives'), (vect, 'negatives'),
                             # (tfidf, 'job_title'), (tfidf, 'summary'), (tfidf, 'positive'), (tfidf, 'negatives'),
                             n_jobs=-1,
                             verbose=True)

clf = Pipeline(
    steps=[("preprocessor", ct), ("classifier", LogisticRegression(max_iter=100))]
)

# Training the model
clf.fit(X_train, y_train)

# Train Inference
y_pred = clf.predict(X_train)

# Train metric
train_f1 = f1_score(y_train, y_pred, average='micro')
print('F1 micro:', train_f1.round(2))
print('F1 macro:', f1_score(y_train, y_pred, average='macro').round(2))
print('F1 weighted:', f1_score(y_train, y_pred, average='weighted').round(2))

# Train Inference
y_test_pred = clf.predict(X_test.drop(columns=cols_to_drop))

# test_f1 = f1_score(y_train, y_pred, average='micro')
# print('F1 micro:', test_f1.round(2))
# print('F1 macro:', f1_score(y_test, y_test_pred, average='macro').round(2))
# print('F1 weighted:', f1_score(y_test, y_test_pred, average='weighted').round(2))

# Save test predictions
# print(len(X_test), len(y_test_pred))
# X_test['overall'] = y_test_pred
pd.DataFrame({"ID": X_test["ID"], "overall": y_test_pred.astype("int")}).to_csv("NLP_Data/submission.csv", index=False)
