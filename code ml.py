#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:26:24 2026

@author: annalenamr
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Excel-Datei laden
df = pd.read_excel("sample_python.xlsx")

# Spalten anzeigen
print("Vorhandene Spalten:")
print(df.columns.tolist())

# Zielvariable definieren
X = df.drop("Order_Conversion", axis=1)
y = df["Order_Conversion"]

# Logistische Regression trainieren
model = LogisticRegression(max_iter=5000)
model.fit(X, y)

print("Modell erfolgreich trainiert.")

# Conversion-Wahrscheinlichkeiten berechnen
wahrscheinlichkeiten = model.predict_proba(X)

# Wahrscheinlichkeit für Conversion (= Klasse 1)
df["Conversion_Wahrscheinlichkeit"] = wahrscheinlichkeiten[:, 1]


# Entscheidung treffen
def entscheidung(wahrscheinlichkeit):
    if wahrscheinlichkeit >= 0.60:
        return "Sample senden"
    elif wahrscheinlichkeit >= 0.40:
        return "Prüfen"
    else:
        return "Kein Sample"

df["Empfehlung"] = df["Conversion_Wahrscheinlichkeit"].apply(entscheidung)

# Verteilung anzeigen
print(df["Empfehlung"].value_counts())

# Ergebnisse anzeigen
print("\nErgebnisse:")
print(
    df[
        [
            "Conversion_Wahrscheinlichkeit",
            "Empfehlung"
        ]
    ].head(20)
)

# Datei speichern
df.to_excel("Ergebnis_ML.xlsx", index=False)

print("\nFertig! Die Datei 'Ergebnis_ML.xlsx' wurde erstellt.")

conversionen = df[df["Order_Conversion"] == 1]

print(conversionen.select_dtypes(include=["number"]).sum().sort_values(ascending=False).head(20))
koeffizienten = pd.DataFrame({
    "Merkmal": X.columns,
    "Gewichtung": model.coef_[0]
})

print(
    koeffizienten.sort_values(
        "Gewichtung",
        ascending=False
    ).head(15)
)

print(
    koeffizienten.sort_values(
        "Gewichtung",
        ascending=True
    ).head(15)
)
import joblib

joblib.dump(model, "conversion_model.pkl")

print("Modell gespeichert")
print(X.columns.tolist())
