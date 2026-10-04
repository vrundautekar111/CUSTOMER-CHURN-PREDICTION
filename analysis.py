import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "dataset/customer_churn.csv"
)

df["total_charges"] = pd.to_numeric(
    df["total_charges"],
    errors="coerce"
)


# Create graph folder
os.makedirs(
    "static/graphs",
    exist_ok=True
)


# --------------------------------------------------
# GRAPH 1 - CHURN DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="churn"
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/churn_distribution.png"
)

plt.close()


# --------------------------------------------------
# GRAPH 2 - CHURN BY CONTRACT
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="contract",
    hue="churn"
)

plt.title(
    "Churn by Contract Type"
)

plt.xlabel(
    "Contract"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "static/graphs/contract_churn.png"
)

plt.close()


# --------------------------------------------------
# GRAPH 3 - TENURE VS CHURN
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="churn",
    y="tenure"
)

plt.title(
    "Tenure vs Churn"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Tenure (Months)"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/tenure_churn.png"
)

plt.close()


# --------------------------------------------------
# GRAPH 4 - MONTHLY CHARGES
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="churn",
    y="monthly_charges"
)

plt.title(
    "Monthly Charges vs Churn"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Monthly Charges"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/monthly_charges.png"
)

plt.close()


# --------------------------------------------------
# GRAPH 5 - INTERNET SERVICE
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="internet_service",
    hue="churn"
)

plt.title(
    "Churn by Internet Service"
)

plt.xlabel(
    "Internet Service"
)

plt.ylabel(
    "Customers"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/internet_churn.png"
)

plt.close()


print(
    "All analysis graphs generated successfully!"
)