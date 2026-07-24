"""Module Demand Forecasting."""
from .data_prep import ModuleDemandFeatureEngineer

try:
    from .regressor import ModuleDemandRegressor
except ImportError:
    ModuleDemandRegressor = None

try:
    from .classifier import ModuleDemandClassifier
except ImportError:
    ModuleDemandClassifier = None

try:
    from .metrics import evaluate_module_demand
except ImportError:
    evaluate_module_demand = None

__all__ = ['ModuleDemandFeatureEngineer', 'ModuleDemandRegressor', 'ModuleDemandClassifier', 'evaluate_module_demand']
