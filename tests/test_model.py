import pytest
import numpy as np
from src.train import eval_metrics

def test_eval_metrics_perfect_prediction():
    """Test metrics when prediction is perfect."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]
    
    acc, prec, rec, f1, auc = eval_metrics(y_true, y_pred, y_proba)
    
    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0
    assert auc == 1.0

def test_eval_metrics_all_wrong():
    """Test metrics when prediction is completely wrong."""
    y_true = [0, 1]
    y_pred = [1, 0]
    y_proba = [0.9, 0.1]
    
    acc, prec, rec, f1, auc = eval_metrics(y_true, y_pred, y_proba)
    
    assert acc == 0.0
    # Precision/Recall might be 0.0 or handle division by zero depending on implementation
    assert auc == 0.0