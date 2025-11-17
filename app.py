
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReFill Hub Dashboard", layout="wide")

st.title("ReFill Hub – Smart Refill Stations Dashboard")

uploaded = st.sidebar.file_uploader("Upload Survey Data (CSV)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Sample Data Preview")
    st.write(df.head())

    # ---------------- HOME PAGE GRAPHS ------------------
    st.header("📊 Data Visualisation")
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in list(num_cols)[:5]:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20)
        st.pyplot(fig)

    for col in list(cat_cols)[:5]:
        st.subheader(f"Count Plot of {col}")
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # ---------------- CLASSIFICATION ------------------
    st.header("🤖 Classification Models")

    target = st.selectbox("Select Target Column", df.columns)
    if st.button("Run Classification"):
        le = LabelEncoder()
        X = df.drop(columns=[target])
        X = X.apply(lambda x: le.fit_transform(x.astype(str)) if x.dtype=='object' else x)
        y = le.fit_transform(df[target].astype(str))

        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        for name, model in models.items():
            st.subheader(name)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            acc = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
            model.fit(X, y)
            preds = model.predict(X)

            st.write(f"Accuracy: {acc:.3f}")
            st.write(f"Precision: {precision_score(y, preds, average='macro'):.3f}")
            st.write(f"Recall: {recall_score(y, preds, average='macro'):.3f}")
            st.write(f"F1 Score: {f1_score(y, preds, average='macro'):.3f}")

            # SAFE ROC CURVE (ONLY FOR BINARY CLASSES)
            if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:,1]
                fpr, tpr, _ = roc_curve(y, prob)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("ROC Curve skipped (Target is not binary).")

    # ---------------- NEW DATA PREDICTION ------------------
    st.header("📥 Add New Data for Prediction")

    if uploaded is not None:
        X = df.drop(columns=[target])
        X_enc = X.apply(lambda x: LabelEncoder().fit(x).transform(x.astype(str)))
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target].astype(str))
        pred_model = RandomForestClassifier().fit(X_enc, y)

        if st.checkbox("Add New Input Row"):
            new_row = {}
            for col in X.columns:
                new_row[col] = st.text_input(f"Enter value for {col}")

            if st.button("Predict New Label"):
                new_df = pd.DataFrame([new_row])
                new_df_enc = new_df.apply(
                    lambda x: LabelEncoder().fit(df[x.name].astype(str)).transform(x.astype(str))
                )
                pred = pred_model.predict(new_df_enc)[0]
                st.success(f"Predicted Label: {le_target.inverse_transform([pred])[0]}")

    # ---------------- K-MEANS ------------------
    st.header("🎯 Customer Segmentation (K-Means)")
    if st.button("Run K-Means"):
        k_data = df.select_dtypes(include=['int64','float64'])
        km = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = km.fit_predict(k_data)
        st.write(df[['Cluster']].head())
        st.write("Cluster Centers:", km.cluster_centers_)

    # ---------------- APRIORI ------------------
    st.header("🔗 Association Rule Mining (Apriori)")
    if st.button("Run Apriori"):
        basket = df.astype(str)
        one_hot = pd.get_dummies(basket)
        freq = apriori(one_hot, min_support=0.1, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.7)
        st.write(rules.head(10))
