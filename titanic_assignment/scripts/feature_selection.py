"""
Part 3: Feature Selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def remove_high_correlation(df, target='Survived', threshold=0.95):
    """Drop features with correlation > threshold (excluding target)."""
    numeric_df = df.drop(columns=[target]).select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    # Upper triangle to avoid duplicate pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"Dropping highly correlated features: {to_drop}")
    df.drop(columns=to_drop, inplace=True)
    return df

def get_feature_importance(df, target='Survived'):
    """Use RandomForest to rank feature importance."""
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print("\nFeature Importances (Random Forest):")
    print(importances)

    # Plot
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    print("\nPlot saved as feature_importances.png")

    return importances

def select_features(input_path, output_path):
    df = pd.read_csv(input_path)

    df = remove_high_correlation(df)
    importances = get_feature_importance(df)

    # Keep features with importance > 0.01
    selected = importances[importances > 0.01].index.tolist()
    selected.append('Survived')

    print(f"\nSelected features: {selected}")
    df_selected = df[selected]

    df_selected.to_csv(output_path, index=False)
    print(f"\nSaved selected features to {output_path}")
    return df_selected

if __name__ == "__main__":
    select_features(
        input_path="data/train_engineered.csv",
        output_path="data/train_selected.csv"
    )
