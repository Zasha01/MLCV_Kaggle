"""Data loading and processing modules."""
from .loader import load_data, get_data_summary
from .features import (
    create_interaction_features,
    apply_label_encoding,
    apply_target_encoding_cv,
    apply_frequency_encoding,
    prepare_datasets
)

__all__ = [
    'load_data',
    'get_data_summary',
    'create_interaction_features',
    'apply_label_encoding',
    'apply_target_encoding_cv',
    'apply_frequency_encoding',
    'prepare_datasets'
]

