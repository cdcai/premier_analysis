#!/usr/bin/env bash
# == Automate running data pre-processing steps

# Construct output paths
out_path="${PWD}/output/"
pkl_path="${out_path}/pkl"
parquet_path="${out_path}/parquet"

mkdir -p ${pkl_path}
mkdir -p ${parquet_path}

# Check if virtualenv present, and if so activate it
venv_path="${PWD}/venv/Scripts/activate"

# Activate virtualenv
if [[ -f "${venv_path}" ]]; then
    echo "Virualenv found, activating"
    source "${venv_path}"
fi

py_location=$(where python || which python)

# Echo the python location
echo "${py_location}"


# Process EHR data to summarized parquet and produce feature dicts
run_feature_extraction() {
    python "$PWD/python/feature_extraction.py"
}
    
# Run features to integers to encode text features to integer representation
run_encoding() {
    # Run DAN
    python "${PWD}/python/feature_tokenization.py"
}

echo "Running feature extraction >>"
run_feature_extraction
echo "Converting text features to integer >>"
run_encoding