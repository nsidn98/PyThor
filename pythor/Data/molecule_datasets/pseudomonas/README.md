# Pseudomonas aeruginosa dataset

Version 1.2

## License

The dataset is released under the Public Domain license, i.e., unrestricted. We
ask that you cite the original, unaltered dataset as 

```
J. Stokes (March 2020), Pseudomonas aeruginosa dataset.
https://www.aicures.mit.edu/data
```

## About the data

The files `train.csv` and `test.csv` contain the SMILES of molecules tested for
activity against Pseudomonas aeruginosa.

To compare your results against our numbers, we have included the 10 splits used
for cross-validation in the folder `train_cv`.

## Submitting your predictions

We ask that you predict the activity of molecules included in the test set
`test.csv`.

A sample submission file is included: `test_predictions_sample.csv`.  The
activity column should be a float between 0 and 1.  Your predictions will be
evaluated against the golden labels to compute an [AUC](https://github.com/yangkevin2/coronavirus_data/blob/master/scripts/evaluate_auc.py).


## Changelog

1.2: CV splits are now in the form of subdirectories.

1.1: Updated `test.csv` and `test_predictions_sample.csv` to accept SMILES as index for predictions.
