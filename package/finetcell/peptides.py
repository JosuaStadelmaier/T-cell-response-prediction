"""
Reading, validating and encoding peptide sequences for the FINE-T model.
"""
import torch
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# The positional encodings of the model are built for 30 tokens,
# which includes the start and end token of a peptide.
MAX_PEPTIDE_LENGTH = 28


def read_peptide_file(file_path):
    """
    Reads peptide sequences from a file.
    Accepts a CSV file with a peptide column ('Epitope Description' or 'peptide'),
    a CSV file with a single column, or a text file with one peptide per line.
    :param file_path: path to the file with peptide sequences
    :return: list of peptide sequences
    """
    if str(file_path).endswith('.csv'):
        df = pd.read_csv(file_path, low_memory=False)
        peptide_columns = [column for column in ['Epitope Description', 'peptide', 'Peptide']
                           if column in df.columns]
        if peptide_columns:
            peptides = df[peptide_columns[0]]
        elif len(df.columns) == 1:
            peptides = df[df.columns[0]]
        else:
            raise ValueError(f'{file_path} has no peptide column. Expected one of '
                             f"'Epitope Description', 'peptide', 'Peptide', or a file with a single column.")
        peptides = peptides.dropna().astype(str).to_list()
    else:
        with open(file_path) as peptide_file:
            peptides = [line.strip() for line in peptide_file
                        if line.strip() and not line.startswith('>')]

    return [peptide.strip().upper() for peptide in peptides]


class PeptideData(object):
    """
    Provides the model input for peptide sequences.

    The FINE-T model only takes the peptide sequence as input, so no MHC alleles,
    peptide sources, presentation scores or labels are needed for predictions.
    This class also provides the attributes that the model reads from a data object.
    """
    def __init__(self, peptides):
        self.amino_acids = AMINO_ACIDS
        self.amino_acid_to_id = {self.amino_acids[i]: i for i in range(len(self.amino_acids))}

        # token indices, identical to the implementation in the code directory
        self.padding_token_idx = len(self.amino_acid_to_id)
        self.prediction_token_idx = len(self.amino_acid_to_id) + 1
        self.pep_start_token_idx = len(self.amino_acid_to_id) + 2
        self.pep_end_token_idx = len(self.amino_acid_to_id) + 2

        self.peptides, self.skipped_peptides = self.validate_peptides(peptides)

        if not self.peptides:
            raise ValueError('No valid peptide sequences to predict on.')

    def validate_peptides(self, peptides):
        """
        Separates the peptides that the model can process from the ones that it cannot.
        :param peptides: list of peptide sequences
        :return: list of valid peptides, list of (peptide, reason) pairs for the remaining peptides
        """
        valid_peptides = []
        skipped_peptides = []
        for peptide in peptides:
            unknown_amino_acids = sorted(set(peptide) - set(self.amino_acid_to_id))
            if not peptide:
                skipped_peptides.append((peptide, 'empty sequence'))
            elif unknown_amino_acids:
                skipped_peptides.append((peptide, f'non-standard amino acids: {", ".join(unknown_amino_acids)}'))
            elif len(peptide) > MAX_PEPTIDE_LENGTH:
                skipped_peptides.append((peptide, f'longer than {MAX_PEPTIDE_LENGTH} amino acids'))
            else:
                valid_peptides.append(peptide)

        return valid_peptides, skipped_peptides

    def get_batch(self, peptides):
        """
        Builds the model input for a list of peptides.
        :param peptides: list of valid peptide sequences
        :return: batch dictionary for the forward pass of the model
        """
        peptide_seq_list = [torch.tensor([self.pep_start_token_idx]
                                         + [self.amino_acid_to_id[amino_acid] for amino_acid in peptide]
                                         + [self.pep_end_token_idx]) for peptide in peptides]
        peptide_seq_padded = pad_sequence(peptide_seq_list, batch_first=True, padding_value=self.padding_token_idx)
        peptide_lengths = torch.tensor([len(peptide_seq) for peptide_seq in peptide_seq_list])
        padding_mask = peptide_seq_padded == self.padding_token_idx

        return {
                'padding_mask': padding_mask,
                'peptide': peptide_seq_padded,
                'peptide_lengths': peptide_lengths,
                }
