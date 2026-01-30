import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_uci_wdbc(path: str = "../data/raw/wdbc.data"):
    """
    Loads UCI raw WDBC (wdbc.data):
    col0 = ID
    col1 = Diagnosis ('M' or 'B')
    col2..31 = 30 features

    Returns:
      X: DataFrame (30 features, with sklearn-compatible names)
      y: Series (0=malignant, 1=benign)  # matches sklearn convention
      ids: Series (ID numbers)
    """
    df = pd.read_csv(path, header=None)

    ids = df.iloc[:, 0].astype(str)
    diagnosis = df.iloc[:, 1].astype(str)

    # Map to match sklearn dataset: 0=malignant, 1=benign
    y = diagnosis.map({"M": 0, "B": 1}).astype(int)

    X = df.iloc[:, 2:].copy()

    # Use sklearn's official feature names so your code stays consistent
    feature_names = load_breast_cancer().feature_names
    X.columns = feature_names

    return X, y, ids
