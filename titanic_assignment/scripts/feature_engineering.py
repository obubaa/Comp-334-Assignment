"""
Part 2: Feature Engineering
"""

import pandas as pd
import numpy as np

def create_family_features(df):
    # FamilySize = SibSp + Parch + 1 (counting the passenger themselves)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone: 1 if travelling alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    return df

def extract_title(df):
    # Extract title from Name column
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
    df['Title'] = df['Title'].str.strip()

    # Group rare titles into 'Other'
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Other')

    return df

def extract_deck(df):
    # Deck comes from the first letter of Cabin
    # Since Cabin was dropped in cleaning, we create a placeholder
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].str[0]
        df['Deck'] = df['Deck'].fillna('Unknown')
    else:
        df['Deck'] = 'Unknown'
    return df

def create_age_groups(df):
    bins = [0, 12, 17, 60, 100]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df

def create_fare_per_person(df):
    # Avoid division by zero (FamilySize is always >= 1)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    return df

def encode_categoricals(df):
    # One-hot encode nominal features
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'AgeGroup'], drop_first=False)

    # Pclass is ordinal — already numeric (1, 2, 3), keep as-is
    return df

def apply_transformations(df):
    # Log transform skewed features (add 1 to avoid log(0))
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Age_log'] = np.log1p(df['Age'])

    return df

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)
    print("Shape before feature engineering:", df.shape)

    df = create_family_features(df)
    df = extract_title(df)
    df = extract_deck(df)
    df = create_age_groups(df)
    df = create_fare_per_person(df)
    df = apply_transformations(df)
    df = encode_categoricals(df)

    # Drop columns not useful for modeling
    df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

    print("Shape after feature engineering:", df.shape)
    print("New columns:", df.columns.tolist())

    df.to_csv(output_path, index=False)
    print(f"\nSaved engineered data to {output_path}")
    return df

if __name__ == "__main__":
    engineer_features(
        input_path="data/train_cleaned.csv",
        output_path="data/train_engineered.csv"
    )
