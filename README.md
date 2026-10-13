# Transfer Learning for T-Cell Response Prediction

This is the repository of the paper ["Transfer Learning for T-Cell Response Prediction"](https://arxiv.org/abs/2403.12117) by Josua Stadelmaier, Brandon Malone, and Ralf Eggeling.

There is also a [video poster presentation](https://youtu.be/USc7sdngFYM?si=WrV1lmG_fduutVKj) of this work.

## Introduction
We study the prediction of T-cell response for specific given peptides, which could, among other applications, be a crucial step towards the development of personalized cancer vaccines. It is a challenging task due to limited, heterogeneous training data featuring a multi-domain structure; such data entail the danger of shortcut learning, where models learn general characteristics of peptide sources, such as the source organism, rather than specific peptide characteristics associated with T-cell response.

Using a transformer model for T-cell response prediction, we show that the danger of inflated predictive performance is not merely theoretical but occurs in practice. Consequently, we propose a domain-aware evaluation scheme. We then study different transfer learning techniques to deal with the multi-domain structure and shortcut learning. We demonstrate a per-source fine tuning approach to be effective across a wide range of peptide sources and further show that our final model outperforms existing state-of-the-art approaches for predicting T-cell responses for human peptides.

## Directory contents
- `code/`: Implementation of FINE-T. The parameters used for generating the figures are specified in `code/params.py`.
- `notebooks/`: Jupyter Notebooks for processing the data set and generating data statistics plots and the result figures of the paper.
- `data/`: To reproduce the cross-validation results, extract the file `data/nested_cv.tar.gz`.
- `saved_models/`: Trained FINE-T models for MHC I and MHC II.
- `model_runs/`: Result files from running the FINE-T model on a list of peptides, as described below.
- `package/`: Source code of the `fine-t` pip package, which provides the predictions of FINE-T.

## Usage via pip package

This repository comes with the FINE-T model fine-tuned on human peptides.
The MHC I and MHC II versions of this model are stored in `saved_models/`.

There are two ways of using these models. The `fine-t` pip package predicts T-cell response
for peptide sequences, which is all that the model needs as input, and it does not evaluate
the predictions. Its source code is in `package/`.
 ```
pip install fine-t
fine-t --peptides model_runs/test_peptides_mhc_I_human.csv --mhc I --source human
 ```

The `--source` parameter needs to be set to `human`, since the models that are fine tuned on
other peptide sources are not part of the package. The `--mhc` parameter selects the model of
the MHC class of the peptides, either `I` or `II`, so provide the peptides of one MHC class
at a time. The file with the peptides is a CSV file with a peptide column
(`Epitope Description` or `peptide`), a CSV file with a single column, or a text file with one
peptide per line. The predictions are written next to it, or to the path given by
`--predictions_file`.

The package also provides the predictions in Python:
 ```python
import finet

predictions = finet.predict(['GILGFVFTL', 'NLVPMVATV'], mhc='I', source='human')
 ```

## Usage of the repository scripts

The scripts of this repository provide the training of the model and the domain-aware
evaluation of the paper, in addition to the predictions.

Creating a conda environment with the needed packages:
 ```
conda env create -f environment.yml
conda activate transfer_tcell
 ```

All commands below are run from the root directory of this repository.

### Running and evaluating the model on labelled peptides

Running the FINE-T model on a list of MHC I peptides:
 ```
python -m code.main --load_model -d FINET --config transformer_pretrained_human_paper --mhc I --eval_data ../model_runs/test_peptides.csv
 ```
For MHC II peptides:
 ```
python -m code.main --load_model -d FINET --config transformer_pretrained_human_paper --mhc II --eval_data ../model_runs/test_peptides.csv
 ```

The list of peptides needs to have the same structure as the example `model_runs/test_peptides.csv`,
which includes the measured T-cell response, since these commands also compute accuracy metrics.
File paths are given relative to the `code` directory.
These commands produce two files in `model_runs` containing the predictions and various accuracy metrics.

### Predicting T-cell response for peptide sequences

When using `--predict`, the trained model only takes the peptide sequence as input, so predictions require nothing but
a list of peptides. No evaluation is performed.
 ```
python -m code.main --predict -d FINET --config transformer_pretrained_human_paper --mhc I --peptides ../model_runs/test_peptides_mhc_I_human.csv
 ```

The peptide file is either a CSV file with a peptide column (`Epitope Description` or `peptide`),
a CSV file with a single column, or a text file with one peptide per line.

In this setting, you need to make sure to only use peptides as input that match the MHC class and peptide source of the pretrained model. Here, we provided only MHC I peptides from human proteins as input.
The predictions are written next to the peptide file, or to the path given by `--predictions_file`.

The predictions of the command above are provided as a reference in
`model_runs/predictions_FINET_I_II_pre_I_test_peptides_mhc_I_human.csv`.

### Peptide sequence requirements

All of the commands above need peptides that consist of the 20 standard amino acids and that are
at most 28 amino acids long, which is the longest peptide that the positional encodings of the
model support. Peptides that contain other characters, such as `X`, and peptides that are longer
are printed with the reason for skipping them when using `--predict`, and they are left out of
the predictions.

## License

The code of this repository and the `fine-t` package are licensed under the
GNU Affero General Public License v3.0, see `LICENSE`.
