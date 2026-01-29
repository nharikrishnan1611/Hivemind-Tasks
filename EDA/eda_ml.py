# ==========================================
# Exploratory Data Analysis + ML Preprocessing (Function-based)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 1. Load Dataset
def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset loaded successfully")
    print(df.head())
    print("\n=================================\n")
    return df


# 2. Data Type Detection
def detect_data_types(df):
    print("Data Types:")
    print(df.dtypes)

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)
    print("\n=================================\n")

    return numerical_cols, categorical_cols


# 3. Summary Statistics
def summary_statistics(df, numerical_cols):
    print("Summary Statistics (Numerical Columns):")
    print(df[numerical_cols].describe())
    print("\n=================================\n")


# 4. Missing Values Analysis
def missing_value_analysis(df):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    print("Missing Values Count:")
    print(missing_count)

    print("\nMissing Values Percentage:")
    print(missing_percent)
    print("\n=================================\n")


# 5. Top 15 Categories
def top_categories(df):
    top_cat = df["category"].value_counts().head(15)
    print("Top Categories:")
    print(top_cat)
    print("\n=================================\n")
    return top_cat


# 6. Visualizations
def create_visualizations(df, top_cat):
    # Histogram
    plt.figure()
    df["price"].hist()
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.savefig("plots/histogram.png")
    plt.close()

    # Bar chart
    plt.figure()
    top_cat.plot(kind="bar")
    plt.title("Top Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.savefig("plots/bar_chart.png")
    plt.close()

    # Box plot
    plt.figure()
    df.boxplot(column="sales")
    plt.title("Sales Box Plot")
    plt.savefig("plots/boxplot.png")
    plt.close()

    print("Plots saved in plots/ folder")
    print("\n=================================\n")


# 7. Feature Scaling
def scale_features(df, numerical_cols):
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Numerical features scaled")
    print("\n=================================\n")
    return df


# 8. Encode Categorical Variables
def encode_categorical(df, categorical_cols):
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    print("Categorical variables encoded")
    print("\n=================================\n")
    return df_encoded


# 9. Train-Test Split
def split_data(df_encoded):
    X = df_encoded.drop("sales", axis=1)
    y = df_encoded["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train-Test split completed")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("\n=================================\n")

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    df = load_dataset("ecommerce.csv")
    numerical_cols, categorical_cols = detect_data_types(df)
    summary_statistics(df, numerical_cols)
    missing_value_analysis(df)
    top_cat = top_categories(df)
    create_visualizations(df, top_cat)
    df = scale_features(df, numerical_cols)
    df_encoded = encode_categorical(df, categorical_cols)
    split_data(df_encoded)

    print("ML-ready dataset prepared successfully")


# Run the program
if __name__ == "__main__":
    main()