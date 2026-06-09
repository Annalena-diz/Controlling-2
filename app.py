#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:25:22 2026

@author: annalenamr
"""
import streamlit as st
import pandas as pd
import joblib

# Modell laden
model = joblib.load("conversion_model.pkl")

st.title("Sample Decision Tool")

# Eingaben

country = st.selectbox(
    "Land auswählen",
    [
        "USA","UK","Italy","Belgium","Romania",
        "Australia","India","Canada","Israel",
        "Poland","South_Africa","Brazil",
        "UAE","China"
    ]
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
        "Table_Tufted"
    ]
)

shape = st.selectbox(
    "Form",
    ["REC", "Round", "Square"]
)

qty = st.number_input(
    "Quantity Required",
    min_value=1,
    value=1
)

area = st.number_input(
    "AreaFt",
    min_value=1.0,
    value=100.0, step=10.0
)

if st.button("Berechnen"):

    daten = {
        "USA":0,
        "UK":0,
        "Italy":0,
        "Belgium":0,
        "Romania":0,
        "Australia":0,
        "India":0,
        "Canada":0,
        "Israel":0,
        "Poland":0,
        "South_Africa":0,
        "Brazil":0,
        "UAE":0,
        "China":0,

        "QtyRequired":qty,

        "Hand_Tufted":0,
        "Durry":0,
        "Double_Back":0,
        "Handwoven":0,
        "Knotted":0,
        "Jacquard":0,
        "Handloom":0,
        "Gun_Tufted":0,
        "Indo_Tibbetan":0,
        "Power_Loom_Jacquard":0,
        "Table_Tufted":0,

        "REC":0,
        "Round":0,
        "Square":0,

        "AreaFt":area
    }

    daten[country] = 1
    daten[item] = 1
    daten[shape] = 1

    eingabe = pd.DataFrame([daten])

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
        f"{wahrscheinlichkeit*100:.1f}%"
    )


    st.success(empfehlung)