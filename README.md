# Transfer Learning for T-Cell Response Prediction

This is the repository of the paper ["Transfer Learning for T-Cell Response Prediction"](https://arxiv.org/abs/2403.12117) by Josua Stadelmaier, Brandon Malone, and Ralf Eggeling.

There is also a [video poster presentation](https://youtu.be/USc7sdngFYM?si=WrV1lmG_fduutVKj) of this work.

## Introduction
We study the prediction of T-cell response for specific given peptides, which could, among other applications, be a crucial step towards the development of personalized cancer vaccines. It is a challenging task due to limited, heterogeneous training data featuring a multi-domain structure; such data entail the danger of shortcut learning, where models learn general characteristics of peptide sources, such as the source organism, rather than specific peptide characteristics associated with T-cell response.

Using a transformer model for T-cell response prediction, we show that the danger of inflated predictive performance is not merely theoretical but occurs in practice. Consequently, we propose a domain-aware evaluation scheme. We then study different transfer learning techniques to deal with the multi-domain structure and shortcut learning. We demonstrate a per-source fine tuning approach to be effective across a wide range of peptide sources and further show that our final model outperforms existing state-of-the-art approaches for predicting T-cell responses for human peptides.

## Usage

This repository comes with the FINE-T model fine-tuned on human peptides.
The MHC I and MHC II versions of this model are stored in `saved_models/`.

Creating a conda environment with the needed packages:
 ```
conda env create -f environment.yml
 ```

Running the FINE-T model on a list of MHC I peptides:
 ```
python main.py --load_model -d FINET --config transformer_pretrained_human_paper --mhc I --eval_data ../model_runs/test_peptides.csv
 ```
For MHC II peptides:
 ```
python main.py --load_model -d FINET --config transformer_pretrained_human_paper --mhc II --eval_data ../model_runs/test_peptides.csv
 ```

The list of peptides needs to have the same structure as the example `model_runs/test_peptides.csv`.
These commands produce two files in `model_runs` containing the predictions and various accuracy metrics.
