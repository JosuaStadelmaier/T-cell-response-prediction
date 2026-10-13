"""
Predicting T-cell response for peptide sequences with the FINE-T model.
"""
import os

import pandas as pd
import torch

from finetcell import models
from finetcell.peptides import PeptideData

MHC_CLASSES = ('I', 'II')

# FINE-T is fine-tuned per peptide source, and this package provides the human models.
PEPTIDE_SOURCES = ('human',)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')


def model_file(mhc='I', source='human'):
    """
    Provides the path of a model that comes with this package.
    :param mhc: MHC class of the peptides, 'I' or 'II'
    :param source: peptide source that the model is fine-tuned on, which needs to be 'human'
    :return: path of the saved model
    """
    if mhc not in MHC_CLASSES:
        raise ValueError(f'mhc needs to be one of {MHC_CLASSES}, but it is {mhc!r}.')
    if source not in PEPTIDE_SOURCES:
        raise ValueError(f'source needs to be one of {PEPTIDE_SOURCES}, but it is {source!r}. '
                         f'The models that are fine-tuned on other peptide sources '
                         f'are not part of this package.')
    return os.path.join(MODEL_DIR, f'FINET_I_II_pre_{mhc}_0_1.tar')


def load_checkpoint(file_path):
    """
    Loads a saved model.

    The saved models contain the parameters of their configuration, so they cannot be loaded
    with weights_only=True, which torch uses by default since version 2.6.
    :param file_path: path of the saved model
    :return: dictionary with the state and the configuration of the saved model
    """
    try:
        return torch.load(file_path, map_location='cpu', weights_only=False)
    except TypeError:
        # torch versions before 1.13 do not have the weights_only argument
        return torch.load(file_path, map_location='cpu')


class Predictor(object):
    """
    Predicts T-cell response for peptide sequences with a saved FINE-T model.
    """
    def __init__(self, peptides, mhc='I', source='human', verbose=True):
        saved_model = load_checkpoint(model_file(mhc, source))
        params = saved_model['params']

        if params['model'] != 'PeptideTransformer':
            raise ValueError(f'Predictions for peptide sequences are only supported for '
                             f'PeptideTransformer, but the saved model is a {params["model"]} model.')
        if params['mhc_presentation_feature']:
            raise ValueError('The saved model uses MHC presentation scores as an additional input, '
                             'so peptide sequences alone are not sufficient for predictions.')

        self.params = params
        self.mhc = mhc
        self.source = source
        self.data = PeptideData(peptides)
        self.model = models.PeptideTransformer(params, self.data)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.model.eval()

        if verbose:
            for peptide, reason in self.data.skipped_peptides:
                print(f'Skipping peptide {peptide}: {reason}')

    def predict(self, batch_size=500):
        """
        Computes the T-cell response predictions.
        :param batch_size: number of peptides per forward pass
        :return: DataFrame with the peptides and their predictions
        """
        peptides = self.data.peptides
        predictions = []
        with torch.no_grad():
            for batch_start in range(0, len(peptides), batch_size):
                batch = self.data.get_batch(peptides[batch_start:batch_start + batch_size])
                batch_predictions, _, _ = self.model(batch)
                predictions.append(torch.sigmoid(batch_predictions[:, 0]))

        return pd.DataFrame({'Epitope Description': peptides,
                             'Prediction': torch.cat(predictions).numpy()})


def predict(peptides, mhc='I', source='human', batch_size=500, verbose=True):
    """
    Predicts the T-cell response for peptide sequences.

    The peptides need to match the MHC class and the peptide source of the model,
    so only use human peptides of the given MHC class.
    Peptides that the model cannot process are left out.

    :param peptides: peptide sequence or list of peptide sequences
    :param mhc: MHC class of the peptides, 'I' or 'II'
    :param source: peptide source that the model is fine-tuned on, which needs to be 'human'
    :param batch_size: number of peptides per forward pass
    :param verbose: print the peptides that the model cannot process
    :return: DataFrame with the peptides and their predictions
    """
    if isinstance(peptides, str):
        peptides = [peptides]
    peptides = [str(peptide).strip().upper() for peptide in peptides]
    return Predictor(peptides, mhc, source, verbose=verbose).predict(batch_size=batch_size)
