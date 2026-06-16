#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainiert eine logistische Regression für Order Conversion.

Enthaltene Änderungen:
1. Preprocessing wie besprochen:
   - Canada, Romania, Israel, Australia, Poland, South_Africa, Brazil, UAE, China -> Other_Countries
   - Gun_Tufted, Table_Tufted, Indo_Tibbetan -> Other_Items
2. Logistic Regression OHNE class_weight="balanced"
3. Ausgabe der echten Conversion Rate
4. Ausgabe der vorhergesagten Wahrscheinlichkeiten auf den Trainingsdaten
5. Cross Validation
6. Speichern als conversion_model.pkl
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Einstellungen
# =========================

BASE_DIR = Path(__file__).resolve().parent

# Passe den Namen an, falls deine Excel-Datei anders heißt.
EXCEL_FILE = BASE_DIR / "Book - Kopie.xlsx"

MODEL_FILE = BASE_DIR / "conversion_model.pkl"

TARGET_COL = "Order_Conversion"

OTHER_COUNTRY_COLS = [
    "Canada",
    "Romania",
    "Israel",
    "Australia",
    "Poland",
    "South_Africa",
    "Brazil",
    "UAE",
    "China",
]

OTHER_ITEM_COLS = [
    "Gun_Tufted",
    "Table_Tufted",
    "Indo_Tibbetan",
]


# =========================
# Funktionen
# =========================

def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Bereitet die Trainingsdaten exakt so auf, wie es später auch die Streamlit-App erwartet.
    """

    if TARGET_COL not in df.columns:
        raise ValueError(f"Die Zielspalte '{TARGET_COL}' wurde nicht gefunden.")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL]).copy()

    # Länder zu Other_Countries zusammenfassen
    existing_country_cols = [col for col in OTHER_COUNTRY_COLS if col in X.columns]

    if existing_country_cols:
        X["Other_Countries"] = X[existing_country_cols].max(axis=1)
        X = X.drop(columns=existing_country_cols)
    else:
        X["Other_Countries"] = 0

    # Items zu Other_Items zusammenfassen
    existing_item_cols = [col for col in OTHER_ITEM_COLS if col in X.columns]

    if existing_item_cols:
        X["Other_Items"] = X[existing_item_cols].max(axis=1)
        X = X.drop(columns=existing_item_cols)
    else:
        X["Other_Items"] = 0

    # Alle Spalten numerisch machen
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fehlende Werte auffüllen
    X = X.fillna(0)

    # Binäre Spalten sauber auf int setzen
    for col in X.columns:
        values = set(X[col].dropna().unique())
        if values.issubset({0, 1, 0.0, 1.0}):
            X[col] = X[col].astype(int)

    return X, y


# =========================
# 1. Daten laden
# =========================

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Excel-Datei nicht gefunden: {EXCEL_FILE}\n"
        "Lege die Excel-Datei in denselben Ordner wie dieses Skript "
        "oder passe EXCEL_FILE im Skript an."
    )

df = pd.read_excel(EXCEL_FILE)

print("=" * 70)
print("DATEN GELADEN")
print("=" * 70)
print(f"Datei: {EXCEL_FILE}")
print(f"Shape original: {df.shape}")
print(f"Spalten original: {list(df.columns)}")


# =========================
# 2. Preprocessing
# =========================

X, y = preprocess_dataframe(df)

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)
print(f"Shape nach Preprocessing: {X.shape}")
print(f"Features nach Preprocessing: {list(X.columns)}")

print("\nLabel-Verteilung:")
print(y.value_counts().sort_index())

print("\nDurchschnittliche echte Conversion Rate:")
print(f"{y.mean():.4f} = {y.mean() * 100:.2f}%")


# =========================
# 3. Modellpipeline
# =========================

# Wichtig:
# Kein class_weight="balanced", weil predict_proba sonst häufig
# zu hohe, schlecht interpretierbare Wahrscheinlichkeiten ausgibt.
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )),
    ]
)


# =========================
# 4. Cross Validation
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION ERGEBNISSE")
print("=" * 70)

for metric in scoring.keys():
    scores = cv_results[f"test_{metric}"]
    print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")


# =========================
# 5. Cross-validated Wahrscheinlichkeiten prüfen
# =========================

cv_probas = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv,
    method="predict_proba"
)[:, 1]

print("\n" + "=" * 70)
print("CHECK: CROSS-VALIDATED PREDICTED PROBABILITIES")
print("=" * 70)
print("Diese Werte zeigen, ob die Modell-Wahrscheinlichkeiten realistisch verteilt sind.")
print(pd.Series(cv_probas).describe())

print("\nVergleich:")
print(f"Echte Conversion Rate:                  {y.mean():.4f} = {y.mean() * 100:.2f}%")
print(f"Durchschnittliche Modell-Wahrscheinlichkeit: {cv_probas.mean():.4f} = {cv_probas.mean() * 100:.2f}%")

if cv_probas.mean() > y.mean() * 1.5:
    print("\nWARNUNG:")
    print("Die durchschnittliche Modell-Wahrscheinlichkeit ist deutlich höher als die echte Conversion Rate.")
    print("Dann sollte man später ggf. kalibrieren oder den Threshold anpassen.")


# =========================
# 6. Finales Modell trainieren
# =========================

pipeline.fit(X, y)

train_probas = pipeline.predict_proba(X)[:, 1]
train_predictions = pipeline.predict(X)

print("\n" + "=" * 70)
print("CHECK: TRAINING PREDICTED PROBABILITIES")
print("=" * 70)
print(pd.Series(train_probas).describe())

print("\nConfusion Matrix auf Trainingsdaten:")
print(confusion_matrix(y, train_predictions))

print("\nClassification Report auf Trainingsdaten:")
print(classification_report(y, train_predictions))


# =========================
# 7. Koeffizienten anzeigen
# =========================

model = pipeline.named_steps["model"]

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0],
}).sort_values(by="Coefficient", ascending=False)

print("\n" + "=" * 70)
print("WICHTIGSTE POSITIVE KOEFFIZIENTEN")
print("=" * 70)
print(coefficients.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("WICHTIGSTE NEGATIVE KOEFFIZIENTEN")
print("=" * 70)
print(coefficients.tail(10).to_string(index=False))


# =========================
# 8. Modell speichern
# =========================

joblib.dump(pipeline, MODEL_FILE)

print("\n" + "=" * 70)
print("MODELL GESPEICHERT")
print("=" * 70)
print(f"Modell wurde gespeichert als: {MODEL_FILE}")
