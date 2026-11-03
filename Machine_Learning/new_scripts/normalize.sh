python normalize.py \
  -i "mixture_formatted_pure_shift.csv" \
  -o real_spectra_normalized_0ppm.csv \
  --summary-out real_spectra_normalization_summary.csv \
  --ref-ppm 0.0 \
  --window 0.01 \
  --target-integral 0.5 \
  --no-divide-ppm-by-1000
