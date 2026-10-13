"""
FINE-T, a transformer model for T-cell response prediction.

This package predicts T-cell response for peptide sequences with the models of the paper
"Transfer Learning for T-Cell Response Prediction" by Josua Stadelmaier, Brandon Malone,
and Ralf Eggeling (https://arxiv.org/abs/2403.12117).

Training the model and reproducing the domain-aware evaluation of the paper is done with
the scripts of https://github.com/JosuaStadelmaier/T-cell-response-prediction.
"""
from finet.api import MHC_CLASSES, PEPTIDE_SOURCES, Predictor, model_file, predict
from finet.peptides import read_peptide_file

__version__ = '1.0'

__all__ = [
    'MHC_CLASSES',
    'PEPTIDE_SOURCES',
    'Predictor',
    'model_file',
    'predict',
    'read_peptide_file',
]
