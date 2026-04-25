"""
Part 1: Data Cleaning
"""

import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def handle_missing_values(df):
    # Age: impute with median (robust to outliers)
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)

    # Fare: impute with median
    fare_median = df['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)

    # Embarked: impute with mode (most frequent port)
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

    # Cabin: too many missing (~77%), drop the column
    df.drop(columns=['Cabin'], inplace=True, errors='ignore')

    return df

def handle_outliers(df):
    # Cap Fare at 99th percentile to reduce extreme outliers
    fare_cap = df['Fare'].quantile(0.99)
    df['Fare'] = df['Fare'].clip(upper=fare_cap)

    # Cap Age at 99th percentile
    age_cap = df['Age'].quantile(0.99)
    df['Age'] = df['Age'].clip(upper=age_cap)

    return df

def fix_consistency(df):
    # Standardize Sex column to lowercase
    df['Sex'] = df['Sex'].str.lower().str.strip()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df

def clean_data(input_path, output_path):
    df = load_data(input_path)
    print("Original shape:", df.shape)
    print("\nMissing values before cleaning:\n", df.isnull().sum())

    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = fix_consistency(df)

    print("\nMissing values after cleaning:\n", df.isnull().sum())
    print("\nCleaned shape:", df.shape)

    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")
    return df

if __name__ == "__main__":
    clean_data(
        input_path="data/train.csv",
        output_path="data/train_cleaned.csv"
    )
