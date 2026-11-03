#python predict_cnn.py \
#  -i reversed.csv \
#  -b cnn_nmr_bundle.joblib \
#  --keras-model cnn_nmr_model.keras \
#  -o cnn_predictions_real_spectra.csv \
#  --clip-negative
python predict_dnn.py \
  -i real_spectra_normalized_0ppm.csv \
  -b dense_nmr_bundle.joblib \
  --keras-model dense_nmr_model.keras \
  -o dense_predictions_real_spectra.csv \
  --clip-negative
  #-i pure_shift_mixtures_formatted_run5-1.csv \
