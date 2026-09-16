import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

st.title("Breast Cancer PCA Explorer")

# Load the dataset
data = load_breast_cancer(as_frame=True)
df = data.frame

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.subheader("Correlation Heatmap")

numeric_data = data.data

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    numeric_data.corr(),
    cmap="vlag",
    center=0,
    ax=ax
)

st.pyplot(fig)

st.subheader("PCA Projection")

# Standardize the numeric variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_data)

# Run PCA with two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a dataframe for PCA results
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# Add diagnosis category
pca_df["Diagnosis"] = data.target.map({
    0: "Malignant",
    1: "Benign"
})
pca_df["Mean Radius Group"] = pd.cut(
    numeric_data["mean radius"],
    bins=3,
    labels=["Small", "Medium", "Large"]
)

color_by = st.selectbox(
    "Color PCA points by:",
    ["Diagnosis", "Mean Radius Group"]
)

# Plot PCA
fig2, ax2 = plt.subplots(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue=color_by,
    ax=ax2
)

ax2.set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
)
ax2.set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
)

st.pyplot(fig2)
# PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=numeric_data.columns
)

st.subheader("PCA Loadings")
st.dataframe(loadings)
st.subheader("Top Contributors")

top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(5)
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(5)

st.write("Top variables contributing to PC1:")
st.write(top_pc1)

st.write("Top variables contributing to PC2:")
st.write(top_pc2)
st.subheader("Interpretation")

st.write("""
The PCA projection shows a clear separation between benign and malignant
samples, although some overlap remains. PC1 mainly represents variation in
tumor shape and size-related features, with strong contributions from concave
points, concavity, compactness, and perimeter. PC2 captures a different source
of variation and is influenced strongly by fractal-dimension-related features.
Together, PC1 and PC2 summarize major patterns in the original numeric
variables in two dimensions.
""")