#!/bin/bash

# Define directories
OUTPUT_DIR="/path/to/your/predictions"  # Update with your local path
FINAL_CSV="/path/to/your/compartment_predictions.csv"  # Update with your local path

# Merge all CSV files into one
echo "Merging per-slice predictions..."
# Manually add the header to the final CSV
echo "slice_number,patch_index,class_0,class_1,class_2,class_3,class_4,class_5,class_6,class_7" > $FINAL_CSV

# Append all data from CSVs, skipping the first line (header)
tail -n +2 -q $OUTPUT_DIR/predictions_slice_*.csv >> $FINAL_CSV

echo "Merged CSV saved as $FINAL_CSV"
