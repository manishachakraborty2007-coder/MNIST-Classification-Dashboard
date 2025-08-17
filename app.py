import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import random

# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("mnist_train.csv")

df = load_data()
X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------
st.sidebar.title("🚀 Navigation")
page = st.sidebar.selectbox(
    "Select a page",
    ("📊 Data Exploration", "🧠 Model Training + Evaluation")
)

st.title("📌 MNIST Classification Dashboard")

# ===========================================================
# PAGE 1: Data Exploration
# ===========================================================
if page == "📊 Data Exploration":

    tab1, tab2, tab3 = st.tabs(["👁️ Overview", "📈 Distribution", "🖼️ Random Image Sample"])

    # ----------------------------------
    # Overview
    # ----------------------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.write(df.head())
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("Missing values per column:")
        st.write(df.isna().sum())

    # ----------------------------------
    # Distribution plot
    # ----------------------------------
    with tab2:
        st.subheader("Label Distribution")
        label_counts = y.value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(label_counts.index.astype(str), label_counts.values)
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # ----------------------------------
    # Random image sample (with button)
    # ----------------------------------
    with tab3:
        st.subheader("Random Digit Sample")
        if st.button("Show another sample"):
            st.session_state["random_index"] = random.randint(0, len(df) - 1)

        random_index = st.session_state.get("random_index", 0)
        image_row = X.iloc[random_index].values.reshape(28, 28)
        label = y.iloc[random_index]

        fig, ax = plt.subplots()
        ax.imshow(image_row, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
        st.pyplot(fig)

# ===========================================================
# PAGE 2: Model Training + Evaluation
# ===========================================================
elif page == "🧠 Model Training + Evaluation":

    tab1, tab2 = st.tabs(["⚙️ Train Model", "📊 Evaluate Model"])

    # ----------------------------------
    # Tab: Train Model
    # ----------------------------------
    with tab1:
        st.subheader("Model Training")

        test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)
        hidden_layer = st.number_input("Hidden layer size", value=100)

        if st.button("Train MLP Classifier"):
            with st.spinner("Training model..."):

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer,), max_iter=50)
                mlp.fit(X_train, y_train)

                y_pred = mlp.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"✅ Training complete – Accuracy: {acc:.4f}")

                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_pred
                st.session_state["X_test"] = X_test

    # ----------------------------------
    # Tab: Evaluation + Predicted Samples
    # ----------------------------------
    with tab2:
        st.subheader("Model Evaluation")

        if "y_test" in st.session_state and "y_pred" in st.session_state:
            y_test = st.session_state["y_test"]
            y_pred = st.session_state["y_pred"]

            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.text("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig2, ax2 = plt.subplots()
            im = ax2.imshow(cm)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("True")
            ax2.set_title("Confusion Matrix")
            st.pyplot(fig2)

            # --- Prediction preview grid
            st.markdown("### 🔍 Preview Predictions")
            num_images = st.slider("Number of images to preview", 4, 25, 9, step=1)

            # random indices from test set
            indices = np.random.choice(len(y_test), num_images, replace=False)
            n_cols = int(np.ceil(np.sqrt(num_images)))

            fig3 = plt.figure(figsize=(8, 8))
            for i, idx in enumerate(indices, 1):
                image = st.session_state["X_test"].iloc[idx].values.reshape(28, 28)
                true_label = y_test.iloc[idx]
                pred_label = y_pred[idx]

                ax = fig3.add_subplot(n_cols, n_cols, i)
                ax.imshow(image, cmap="gray")
                ax.set_title(f"Pred: {pred_label} | True: {true_label}", fontsize=8)
                ax.axis("off")

            st.pyplot(fig3)
        else:
            st.info("Please train the model first.")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("---")
st.markdown("🔹 *© 2025 MNIST Classification Dashboard. All rights reserved.*")
