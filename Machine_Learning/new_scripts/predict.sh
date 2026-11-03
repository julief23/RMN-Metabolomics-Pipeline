#python predict.py -i pure_shift_mixtures_formatted_run5-1.csv -m pls_nmr_model_WITH_MASK.joblib --clip-negative
python predict.py -i real_spectra_normalized_0ppm.csv -m pls_nmr_model_WITH_MASK.joblib --clip-negative
#python predict.py -i real_norm.csv -m pls_nmr_model_WITH_MASK.joblib --clip-negative
