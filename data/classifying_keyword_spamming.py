import nltk
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Warning: Could not download NLTK punkt: {e}")
    print("You may need to run: python -m nltk.downloader punkt")

from itertools import chain
import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import spacy
import truecase
import xgboost as xgb
import unidecode

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found.")
    print("Please install it using: python -m spacy download en_core_web_sm")
    print("Or run the setup script: python setup.py")
    exit(1)


cwd = os.getcwd()

train = pd.read_csv(os.path.join(cwd, "data/train_set.csv"), sep="\t")[
    ["product_id", "description", "label"]
]
test = pd.read_csv(os.path.join(cwd, "data/test_set.csv"), sep="\t")[
    ["product_id", "description", "label"]
]
df = pd.concat([train, test], axis=0)

df = df.reset_index(drop=True)
df[:2]


def proportion_of_stopwords(description):
    tokens = description.split(" ")
    num_stopwords = len(
        [word for word in tokens if word.lower() in nlp.Defaults.stop_words]
    )
    return float(num_stopwords) / float(len(tokens)) if len(tokens) else 0.0


def average_length_of_word(description):
    tokens = description.split(" ")
    return np.mean([len(word) for word in tokens]) if len(tokens) else 0.0


def proportion_of_numbers(description):
    tokens = description.split(" ")
    num_digits = len([word for word in tokens if word.isdigit()])
    return float(num_digits) / float(len(tokens)) if len(tokens) else 0.0


def normalise_nonascii_chars(input_str):
    return unidecode.unidecode(input_str)


def replace_special_chars(main_string):
    return re.sub(r"[,;@#!\?\+\*\n\-: /]", " ", main_string)


def keep_alphanumeric_chars(string_input):
    return re.sub("[^A-Za-z0-9& ]", "", string_input)


def remove_spaces(string_input):
    return " ".join(string_input.split())


def lemmatize(string_input):
    token_object = nlp(string_input)
    lemmas_list = [
        word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in token_object
    ]
    return " ".join(lemmas_list)


def clean_description(input_str):
    input_str = replace_special_chars(input_str.lower())
    input_str = normalise_nonascii_chars(input_str)
    input_str = keep_alphanumeric_chars(input_str)
    input_str = lemmatize(input_str)
    input_str = remove_spaces(input_str)
    return input_str


df['description'] = df['description'].apply(clean_description)

for i in ['proportion_of_stopwords', 'average_length_of_word', 'proportion_of_numbers']:
    df[i] = df['description'].apply(eval(i))
    
df[:1]    


def drop_digits(s):
    return "".join([i for i in s if not i.isdigit()])


df["description"] = df["description"].apply(drop_digits)
# .ents only stands a chance with capitalized words
df["description_truecase"] = df["description"].apply(truecase.get_true_case)
df["description_nlp"] = df["description_truecase"].apply(nlp)

df["named_entities"] = ""
for i, description_nlp in df["description_nlp"].iteritems():
    named_entities_sets = description_nlp.ents
    named_entities = list(set(chain(*named_entities_sets)))
    df["named_entities"].iloc[i] = " ".join(j.text for j in named_entities)


vectorizer_named_entities = TfidfVectorizer(
    stop_words=nlp.Defaults.stop_words, max_features=500
)
df_tfidf_named_entities = pd.DataFrame(
    vectorizer_named_entities.fit_transform(df["named_entities"]).todense(),
    columns=vectorizer_named_entities.get_feature_names(),
)
df_tfidf_named_entities.columns = [
    f"{i}_named_entities" for i in df_tfidf_named_entities.columns
]
df = pd.concat([df, df_tfidf_named_entities], axis=1)
df[:10]


vectorizer_description = TfidfVectorizer(
    stop_words=nlp.Defaults.stop_words, max_features=500
)
df_tfidf_description = pd.DataFrame(
    vectorizer_description.fit_transform(df["description"]).todense(),
    columns=vectorizer_description.get_feature_names(),
)
df_tfidf_description.columns = [
    f"{i}_description" for i in df_tfidf_description.columns
]
df = pd.concat([df, df_tfidf_description], axis=1)
df[:1]


df = df.drop(
    ["description", "description_truecase", "description_nlp", "named_entities"], axis=1
)
df[:1]


X_train = df.drop("label", axis=1)
y_train = df["label"]
X_test = df[df["product_id"].isin(list(test["product_id"]))].drop("label", axis=1)
y_test = df[df["product_id"].isin(list(test["product_id"]))]["label"]


param_dist = {"objective": "binary:logistic", "n_estimators": 2}

clf = xgb.XGBClassifier(**param_dist)

clf.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="logloss",
    verbose=True,
)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {(accuracy * 100.0):.2f}%")


features = pd.DataFrame(
    zip(X_train.columns, clf.feature_importances_),
    columns=["feature_name", "feature_importance"],
)
features.sort_values("feature_importance", ascending=False)
