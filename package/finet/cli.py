"""
Command line interface for predicting T-cell response for peptide sequences.
"""
import argparse
import os

from finet.api import MHC_CLASSES, PEPTIDE_SOURCES, Predictor
from finet.peptides import read_peptide_file


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='fine-t',
        description='Predict T-cell response for peptide sequences with the FINE-T model.')
    parser.add_argument('--peptides', required=True,
                        help='File with peptide sequences: a CSV file with a peptide column, '
                             'a CSV file with a single column, or one peptide per line')
    parser.add_argument('--mhc', required=True, choices=list(MHC_CLASSES),
                        help='MHC class of the peptides')
    parser.add_argument('--source', required=True, choices=list(PEPTIDE_SOURCES),
                        help='Peptide source that the model is fine-tuned on')
    parser.add_argument('--predictions_file', default=None,
                        help='Output file, by default next to the file with the peptides')
    args = parser.parse_args(argv)

    peptide_file = os.path.abspath(args.peptides)
    peptides = read_peptide_file(peptide_file)
    predictions_df = Predictor(peptides, args.mhc, args.source).predict()

    if args.predictions_file is not None:
        predictions_file_path = os.path.abspath(args.predictions_file)
    else:
        peptide_file_name = os.path.splitext(os.path.basename(peptide_file))[0]
        predictions_file_path = os.path.join(os.path.dirname(peptide_file),
                                             f'predictions_{args.source}_mhc_{args.mhc}_'
                                             f'{peptide_file_name}.csv')
    predictions_df.to_csv(predictions_file_path, index=False)
    print(f'Wrote {len(predictions_df)} predictions to {predictions_file_path}')


if __name__ == '__main__':
    main()
