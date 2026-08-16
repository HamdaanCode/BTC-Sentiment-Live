#!/bin/bash

# Fast extraction of Sept-Nov 2021 using awk
# Much faster than pandas for simple date filtering

INPUT="bitcoin-tweets-2021.csv"
OUTPUT="bitcoin_sept_nov_2021_NEW_EXTRACTED.csv"

echo "FAST EXTRACTION: September-November 2021"
echo ""
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Date range: 2021-09-01 to 2021-11-30"
echo ""
echo "Extracting tweets..."
echo ""

# Extract header
head -1 "$INPUT" > "$OUTPUT"

# Extract Sept-Nov 2021 tweets
# Looking for datetime starting with 2021-09, 2021-10, or 2021-11
awk -F',' '$1 ~ /^2021-(09|10|11)-/' "$INPUT" >> "$OUTPUT"

echo ""
echo " Extraction complete!"
echo ""

# Count results
TOTAL_LINES=$(wc -l < "$OUTPUT")
DATA_LINES=$((TOTAL_LINES - 1))

echo "Results:"
echo "  Total lines (including header): $(printf '%,d' $TOTAL_LINES)"
echo "  Data rows: $(printf '%,d' $DATA_LINES)"
echo ""

# Show sample
echo "First 5 tweets:"
head -6 "$OUTPUT" | tail -5
echo ""
echo " Done! File saved to: $OUTPUT"
