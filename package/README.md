# FINE-T

Predict T-cell response for peptide sequences with FINE-T, the fine-tuned transformer model
of the paper
["Transfer learning for T-cell response prediction"](https://link.springer.com/article/10.1186/s12859-026-06492-2)
by Josua Stadelmaier, Brandon Malone, and Ralf Eggeling, published in BMC Bioinformatics.

FINE-T is fine-tuned per peptide source and MHC class, and this package provides the models
for human peptides of MHC class I and MHC class II. The models only take the peptide sequence
as input.

For training the model and for reproducing the domain-aware evaluation of the paper, use the
scripts of the [repository](https://github.com/JosuaStadelmaier/T-cell-response-prediction).

## Installation

```
pip install fine-t
```

## Usage

```
fine-t --peptides peptides.csv --mhc I --source human
```

The file with the peptides is a CSV file with a peptide column (`Epitope Description` or
`peptide`), a CSV file with a single column, or a text file with one peptide per line.
The predictions are written next to it, or to the path given by `--predictions_file`.

`--source` currently only accepts `human`, since the models that are fine-tuned on other
peptide sources are not part of this package.

`--mhc` is either `I` or `II` and selects the model trained specifically for the respective
MHC class. So use as input the peptides of one MHC class at a time.

From Python:

```python
import finet

predictions = finet.predict(['GILGFVFTL', 'NLVPMVATV'], mhc='I', source='human')
```

`finet.predict` returns a DataFrame with the peptides and their predictions.

Peptides need to consist of the 20 standard amino acids and they can be at most 28 amino
acids long, which is the longest peptide that the positional encodings of the model support.
Peptides that contain other characters, such as `X`, and peptides that are longer are
printed with the reason for skipping them, and they are left out of the predictions.

## Tests

The tests check that the package reproduces the predictions of the repository scripts.
They are part of the source code, so run them from a clone of the repository:

```
pip install -e .[test]
pytest tests
```

## License

FINE-T is licensed under the GNU Affero General Public License v3.0, see `LICENSE`.

## Citation

Stadelmaier, J., Malone, B. & Eggeling, R. Transfer learning for T-cell response prediction.
*BMC Bioinformatics* **27**, 124 (2026). <https://link.springer.com/article/10.1186/s12859-026-06492-2>
