"""
Tests that the package reproduces the predictions of the FINE-T models.

The expected predictions are the predictions of the scripts in the code directory of
https://github.com/JosuaStadelmaier/T-cell-response-prediction, for the peptides of
model_runs/test_peptides_mhc_I_human.csv and the corresponding MHC II peptides.
The tolerance allows for the differences between PyTorch versions, which are far smaller
than the differences that a wrong model or a wrong encoding of the peptides would cause.
"""
import os

import pytest

import finetcell
from finetcell.api import load_checkpoint

TOLERANCE = 1e-5

# predictions of the repository scripts with --mhc I
MHC_I_PREDICTIONS = {
    'QYSWFVNGTF': 0.50431144,
    'YLLKPVQRI': 0.5793473,
    'MIMQGGFSV': 0.4083343,
    'KLLMVLMLA': 0.07899021,
    'TINPQVSKT': 0.4030894,
}

# predictions of the repository scripts with --mhc II
MHC_II_PREDICTIONS = {
    'TSAHLERMKKNMEQTIKDL': 0.12314521,
    'NFFRMVISNPAAT': 0.9866117,
    'NFFRMVISNPA': 0.9999001,
    'KVNFFRMVISNPAATHQD': 0.99998903,
    'FFRMVISNPAATHQDIDFLI': 0.8427059,
}

REFERENCE_FILE = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'model_runs',
                              'predictions_FINET_I_II_pre_I_test_peptides_mhc_I_human.csv')


@pytest.mark.parametrize('mhc,expected', [('I', MHC_I_PREDICTIONS), ('II', MHC_II_PREDICTIONS)])
def test_predictions_of_the_repository_scripts(mhc, expected):
    predictions = finetcell.predict(list(expected), mhc=mhc, source='human', verbose=False)
    predictions = predictions.set_index('Epitope Description')['Prediction']

    assert list(predictions.index) == list(expected)
    for peptide, expected_prediction in expected.items():
        assert predictions[peptide] == pytest.approx(expected_prediction, abs=TOLERANCE)


@pytest.mark.parametrize('mhc', ['I', 'II'])
def test_the_model_of_the_given_mhc_class_is_loaded(mhc):
    params = load_checkpoint(finetcell.model_file(mhc=mhc, source='human'))['params']

    assert params['mhc_class'] == mhc
    assert params['model'] == 'PeptideTransformer'
    assert not params['mhc_presentation_feature']


def test_the_mhc_classes_use_different_models():
    peptides = list(MHC_II_PREDICTIONS)
    predictions_I = finetcell.predict(peptides, mhc='I', source='human', verbose=False)['Prediction']
    predictions_II = finetcell.predict(peptides, mhc='II', source='human', verbose=False)['Prediction']

    assert (predictions_I - predictions_II).abs().max() > 0.1


def test_peptides_are_normalized():
    predictions = finetcell.predict([' qyswfvngtf '], mhc='I', source='human', verbose=False)

    assert predictions['Epitope Description'].to_list() == ['QYSWFVNGTF']
    assert predictions['Prediction'][0] == pytest.approx(MHC_I_PREDICTIONS['QYSWFVNGTF'], abs=TOLERANCE)


def test_peptides_that_the_model_cannot_process_are_left_out():
    peptides = ['QYSWFVNGTF', 'SIINFEXL', 'A' * 29, '']
    predictions = finetcell.predict(peptides, mhc='I', source='human', verbose=False)

    assert predictions['Epitope Description'].to_list() == ['QYSWFVNGTF']


def test_predictions_do_not_depend_on_the_batch_size():
    peptides = list(MHC_I_PREDICTIONS) + list(MHC_II_PREDICTIONS)
    one_batch = finetcell.predict(peptides, mhc='I', source='human', verbose=False)['Prediction']
    single_peptides = finetcell.predict(peptides, mhc='I', source='human', batch_size=1, verbose=False)['Prediction']

    assert (one_batch - single_peptides).abs().max() == pytest.approx(0, abs=TOLERANCE)


@pytest.mark.parametrize('mhc,source', [('III', 'human'), ('i', 'human'), ('I', 'mouse'), ('I', 'Human')])
def test_unavailable_models_are_rejected(mhc, source):
    with pytest.raises(ValueError):
        finetcell.model_file(mhc=mhc, source=source)


def test_no_valid_peptides_is_an_error():
    with pytest.raises(ValueError):
        finetcell.predict(['SIINFEXL'], mhc='I', source='human', verbose=False)


@pytest.mark.skipif(not os.path.exists(REFERENCE_FILE),
                    reason='the reference predictions are only available in the repository')
def test_all_reference_predictions_of_the_repository():
    import pandas as pd

    reference = pd.read_csv(REFERENCE_FILE).set_index('Epitope Description')['Prediction']
    predictions = finetcell.predict(reference.index.to_list(), mhc='I', source='human', verbose=False)
    predictions = predictions.set_index('Epitope Description')['Prediction']

    assert len(predictions) == len(reference)
    assert (reference - predictions).abs().max() == pytest.approx(0, abs=TOLERANCE)
    assert (reference.rank() == predictions.rank()).all()
