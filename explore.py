"""Predict death from brain tumor within 12 months of diagnosis (SEER, 2000-2023).

Steps, in order:
  0. Clean the data and build the 12-month target
  1. Baseline logistic regression on one 80/20 split
  2. Ablation: drop radiation / chemo (immortal time bias check)
  3. Top coefficients
  4. 5-fold cross-validation: logistic regression vs random forest

months, vital and cod are used to BUILD the target and are never model inputs.
Run from the repo root:  python explore.py   (needs data/export.csv, see data/README.md)
"""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/export.csv")
df = df.drop(columns=["RX Summ--Surg Prim Site 2023 (2023+)",
                      "Behavior code ICD-O-3"])

df = df.rename(columns={
    "Age recode with <1 year olds and 90+": "age",
    "Sex": "sex",
    "Year of diagnosis": "year",
    "Histologic Type ICD-O-3": "histology",
    "Combined Summary Stage with Expanded Regional Codes (2004+)": "stage",
    "Radiation recode": "radiation",
    "Chemotherapy recode (yes, no/unk)": "chemo",
    "Survival months": "months",
    "Vital status recode (study cutoff used)": "vital",
    "SEER cause-specific death classification": "cod",
    "Derived Summary Grade 2018 (2018+)": "grade_new",
    "Grade Recode (thru 2017)": "grade_old",
})

# --- merge grade ---
grade_map_new = {
    "Site-specific grade system category (1)": "Grade 1",
    "Site-specific grade system category (2)": "Grade 2",
    "Site-specific grade system category (3)": "Grade 3",
    "Site-specific grade system category (4)": "Grade 4",
    "Well differentiated": "Grade 1",
    "Moderately differentiated": "Grade 2",
    "Poorly differentiated": "Grade 3",
    "Undifferentiated and anaplastic": "Grade 4",
    "High grade": "High/Low (2-tier)",
    "Low grade": "High/Low (2-tier)",
    "Grade cannot be assessed; Unknown": "Unknown",
}
grade_map_old = {
    "Well differentiated; Grade I": "Grade 1",
    "Moderately differentiated; Grade II": "Grade 2",
    "Poorly differentiated; Grade III": "Grade 3",
    "Undifferentiated; anaplastic; Grade IV": "Grade 4",
    "Unknown": "Unknown",
}
df["grade"] = df["grade_new"].map(grade_map_new).fillna(df["grade_old"].map(grade_map_old))

# --- collapse radiation ---
rad_map = {
    "Beam radiation": "Beam",
    "None/Unknown": "None/Unknown",
    "Refused (1988+)": "Refused",
    "Recommended, unknown if administered": "Recommended, unknown",
    "Radiation, NOS  method or source not specified": "Other radiation",
    "Combination of beam with implants or isotopes": "Other radiation",
    "Radioactive implants (includes brachytherapy) (1988+)": "Other radiation",
    "Radioisotopes (1988+)": "Other radiation",
}
df["radiation"] = df["radiation"].map(rad_map)

# --- stage: blank means "not collected yet" (2000-2003) ---
df["stage"] = df["stage"].replace("Blank(s)", "Not collected (pre-2004)")

df = df.drop(columns=["grade_new", "grade_old"])

# --- survival months: text to number, drop rows that aren't numbers ---
df["months"] = pd.to_numeric(df["months"], errors="coerce")
df = df.dropna(subset=["months"])

# --- drop patients who died but the cause is unknown ---
df = df[df["cod"] != "Dead (missing/unknown COD)"]

# --- build the 12-month target ---
died_of_cancer = df["cod"] == "Dead (attributable to this cancer dx)"
df["target"] = np.select(
    [died_of_cancer & (df["months"] <= 12), df["months"] > 12],
    [1, 0],
    default=-1,
)
df = df[df["target"] != -1]

print("Final rows:", len(df))
print("Share who died of cancer within 12 months:", round(df["target"].mean(), 3))

# ---------- MODEL 1 ----------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

df["histology"] = df["histology"].astype(str)   # it's a tumor code, not a quantity
df["year_c"] = df["year"] - 2000

features = ["age", "sex", "year_c", "histology", "stage", "radiation", "chemo", "grade"]
categorical = ["age", "sex", "histology", "stage", "radiation", "chemo", "grade"]

X = df[features]
y = df["target"]

# 80% to learn from, 20% the model never sees
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), categorical)],
    remainder="passthrough")

def report(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print(f"{name}: accuracy {accuracy_score(y_test, pred):.3f}, "
          f"precision {precision_score(y_test, pred, zero_division=0):.3f}, "
          f"recall {recall_score(y_test, pred, zero_division=0):.3f}, "
          f"AUC {roc_auc_score(y_test, prob):.3f}")

print()
report("Always guess survived", Pipeline([("prep", prep), ("clf", DummyClassifier(strategy="most_frequent"))]))
report("Logistic regression  ", Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]))

# ---------- STEP 1: ABLATION (drop treatment columns) ----------
# Same split, same model, same settings. Only the feature list changes.
def auc_for(feature_list):
    cats = [c for c in categorical if c in feature_list]
    p = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), cats)],
        remainder="passthrough")
    m = Pipeline([("prep", p), ("clf", LogisticRegression(max_iter=2000))])
    m.fit(X_train[feature_list], y_train)
    return roc_auc_score(y_test, m.predict_proba(X_test[feature_list])[:, 1])

print("\n--- Ablation: logistic regression, same 80/20 split ---")
auc_full = auc_for(features)
for label, dropped in [("no radiation", ["radiation"]),
                       ("no chemo", ["chemo"]),
                       ("no radiation AND no chemo", ["radiation", "chemo"])]:
    a = auc_for([f for f in features if f not in dropped])
    print(f"{label:28s} AUC {a:.3f}  (change {a - auc_full:+.3f})")
print(f"{'all features':28s} AUC {auc_full:.3f}")


# ---------- STEP 2: TOP COEFFICIENTS ----------
# Positive = pushes predicted risk of dying within 12 months UP; negative = DOWN.
# Each one-hot column is "this patient has this value" (1) or not (0), so a coefficient
# compares that value to the model's baseline. year_c is per year after 2000.
final = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])
final.fit(X_train, y_train)
names = final.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(final.named_steps["clf"].coef_[0], index=names)
coefs.index = coefs.index.str.replace("cat__", "").str.replace("remainder__", "")
coefs = coefs.sort_values()

print("\n--- Top 15 pushing risk UP (log-odds) ---")
print(coefs.tail(15)[::-1].round(2).to_string())
print("\n--- Top 15 pushing risk DOWN (log-odds) ---")
print(coefs.head(15).round(2).to_string())
print("\nTreatment + year coefficients:")
print(coefs[coefs.index.str.startswith(("radiation", "chemo", "year_c"))].round(2).to_string())


# ---------- STEP 3: 5-FOLD CROSS-VALIDATION, LOGISTIC vs RANDOM FOREST ----------
# Instead of one 80/20 split, split into 5 chunks; each chunk takes a turn as the test set.
# The pipeline includes the encoder, so every fold fits it only on that fold's training data.
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic regression": Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]),
    "Random forest": Pipeline([("prep", prep), ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=42))]),
}

print("\n--- 5-fold cross-validated AUC (all 109,956 rows) ---")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"{name:20s} folds {np.round(scores, 3)}  mean {scores.mean():.3f}  std {scores.std():.3f}")
