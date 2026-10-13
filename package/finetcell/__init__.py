"""
FINE-T, a transformer model for T-cell response prediction.

This package predicts T-cell response for peptide sequences with the models of the paper
"Transfer learning for T-cell response prediction" by Josua Stadelmaier, Brandon Malone,
and Ralf Eggeling, published in BMC Bioinformatics
(https://link.springer.com/article/10.1186/s12859-026-06492-2).

Training the model and reproducing the domain-aware evaluation of the paper is done with
the scripts of https://github.com/JosuaStadelmaier/T-cell-response-prediction.
"""
from finetcell.api import MHC_CLASSES, PEPTIDE_SOURCES, Predictor, model_file, predict
from finetcell.peptides import read_peptide_file

__version__ = '1.0'

__all__ = [
    'MHC_CLASSES',
    'PEPTIDE_SOURCES',
    'Predictor',
    'model_file',
    'predict',
    'read_peptide_file',
]
