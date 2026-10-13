#!/usr/bin/env bash
# Downloads the public IDS benchmarks used by scripts/qztna/run_real_data.py into
# data/external/ (git-ignored). Sources are Hugging Face mirrors of the official
# releases; row counts are checked by qztna/data/real_datasets.py on load.
#   UNSW-NB15 official partition: 175,341 training / 82,332 testing records
#   CIC-IDS2017 MachineLearningCVE: 8 day files, 2,830,743 flows
set -euo pipefail
cd "$(dirname "$0")/../data"
mkdir -p external/unsw external/cicids2017
HF=https://huggingface.co/datasets
# The Mireu-Lab mirror names the official files the other way round (its test.csv is the 175k training set).
curl -fsSL -o external/unsw/UNSW_NB15_training-set.csv "$HF/Mireu-Lab/UNSW-NB15/resolve/main/test.csv"
curl -fsSL -o external/unsw/UNSW_NB15_testing-set.csv  "$HF/Mireu-Lab/UNSW-NB15/resolve/main/train.csv"
for f in Monday-WorkingHours Tuesday-WorkingHours Wednesday-workingHours \
         Thursday-WorkingHours-Morning-WebAttacks Thursday-WorkingHours-Afternoon-Infilteration \
         Friday-WorkingHours-Morning Friday-WorkingHours-Afternoon-PortScan Friday-WorkingHours-Afternoon-DDos; do
  curl -fsSL -o "external/cicids2017/$f.csv" "$HF/c01dsnap/CIC-IDS2017/resolve/main/$f.pcap_ISCX.csv"
done
wc -l external/unsw/*.csv
