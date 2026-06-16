#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import pandas as pd
import streamlit as st


# =========================
# Modell laden
# =========================

MODEL_FILE = "conversion_model.pkl"

model = joblib.load(MODEL_FILE)

st.title("Sample Decision Tool")


# =========================
# Mapping wie im Training
# =========================

other_countries = [
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

other_items = [
    "Gun_Tufted",
    "Table_Tufted",
    "Indo_Tibbetan",
]


# =========================
# Erwartete Modell-Features holen
# =========================

try:
    feature_columns = list(model.feature_names_in_)
except AttributeError:
    # Fallback, falls sklearn die Feature-Namen nicht gespeichert hat.
    # Diese Liste entspricht dem Preprocessing aus train_conversion_model.py.
    feature_columns = [
        "UK",
        "Italy",
        "Belgium",
        "India",
        "QtyRequired",
        "Durry",
        "Double_Back",
        "Handwoven",
        "Knotted",
        "Jacquard",
        "Handloom",
        "Power_Loom_Jacquard",
        "Round",
        "Square",
        "AreaFt",
        "Other_Countries",
        "Other_Items",
    ]


# =========================
# Eingaben
# =========================

country = st.selectbox(
    "Land auswählen",
    [
        "USA",
        "UK",
        "Italy",
        "Belgium",
        "Romania",
        "Australia",
        "India",
        "Canada",
        "Israel",
        "Poland",
        "South_Africa",
        "Brazil",
        "UAE",
        "China",
    ],
)

item = st.selectbox(
    "Produkttyp",
    [
        "Hand_Tufted",
        "Durry",
        "Double_Back",
        "Handwoven",
        "Knotted",
        "Jacquard",
        "Handloom",
        "Gun_Tufted",
        "Indo_Tibbetan",
        "Power_Loom_Jacquard",
        "Table_Tufted",
    ],
)

shape = st.selectbox(
    "Form",
    ["REC", "Round", "Square"],
)

qty = st.number_input(
    "Quantity Required",
    min_value=1,
    value=1,
)

area = st.number_input(
    "AreaFt",
    min_value=1.0,
    value=100.0,
    step=10.0,
)


# =========================
# Prediction
# =========================

if st.button("Berechnen"):

    # Alle erwarteten Modell-Spalten mit 0 initialisieren
    daten = {col: 0 for col in feature_columns}

    # Numerische Werte setzen
    if "QtyRequired" in daten:
        daten["QtyRequired"] = qty

    if "AreaFt" in daten:
        daten["AreaFt"] = area

    # Länder setzen:
    # Länder aus der Zusammenfassung werden NICHT einzeln gesetzt,
    # sondern laufen alle über Other_Countries.
    if country in other_countries:
        if "Other_Countries" in daten:
            daten["Other_Countries"] = 1
    else:
        if country in daten:
            daten[country] = 1

    # Items setzen:
    # Gun_Tufted, Table_Tufted, Indo_Tibbetan werden NICHT einzeln gesetzt,
    # sondern laufen alle über Other_Items.
    if item in other_items:
        if "Other_Items" in daten:
            daten["Other_Items"] = 1
    else:
        if item in daten:
            daten[item] = 1

    # Shape setzen:
    # REC ist im Trainingsdatensatz vermutlich Baseline und hat deshalb keine eigene Spalte.
    if shape in daten:
        daten[shape] = 1

    # DataFrame in exakt der Feature-Reihenfolge des Modells
    eingabe = pd.DataFrame([daten])
    eingabe = eingabe[feature_columns]

    wahrscheinlichkeit = model.predict_proba(eingabe)[0][1]

    if wahrscheinlichkeit >= 0.60:
        empfehlung = "✅ Sample senden"
    elif wahrscheinlichkeit >= 0.40:
        empfehlung = "⚠️ Prüfen"
    else:
        empfehlung = "❌ Kein Sample"

    st.subheader("Ergebnis")

    st.metric(
        "Conversion-Wahrscheinlichkeit",
        f"{wahrscheinlichkeit * 100:.1f}%",
    )

    st.success(empfehlung)
