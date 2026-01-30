import pandas as pd

def apply_three_way(p: pd.Series, alpha: float = 0.90, beta: float = 0.10) -> pd.Series:
    """
    Three-way decision based on probability p of class 1 (benign).
    - p >= alpha -> Confirm Benign
    - p <= beta  -> Confirm Malignant
    - otherwise  -> Uncertain
    """
    labels = []
    for val in p:
        if val >= alpha:
            labels.append("Confirm_Benign")
        elif val <= beta:
            labels.append("Confirm_Malignant")
        else:
            labels.append("Uncertain")
    return pd.Series(labels, index=p.index)

def evaluate_three_way(df: pd.DataFrame, alpha: float, beta: float) -> dict:
    """
    df must contain columns: y_true (0/1), p_class1
    Returns: coverage + error rates on non-uncertain cases
    """
    decision = apply_three_way(df["p_class1"], alpha=alpha, beta=beta)
    df = df.copy()
    df["decision"] = decision

    non_uncertain = df[df["decision"] != "Uncertain"]
    coverage = len(non_uncertain) / len(df)

    # Map decisions to predicted class:
    # Confirm_Benign -> 1, Confirm_Malignant -> 0
    pred_class = non_uncertain["decision"].map({
        "Confirm_Benign": 1,
        "Confirm_Malignant": 0
    })

    accuracy_non_uncertain = (pred_class.values == non_uncertain["y_true"].values).mean() if len(non_uncertain) else 0.0

    return {
        "alpha": alpha,
        "beta": beta,
        "coverage": coverage,
        "n_total": len(df),
        "n_non_uncertain": len(non_uncertain),
        "accuracy_non_uncertain": accuracy_non_uncertain
    }
