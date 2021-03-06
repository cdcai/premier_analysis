#!/usr/bin/env bash
# == Automate running all models for the specified outcome

# Check that outcome was passed. Exit if not, or if not allowed.
if [[ $# -eq 0 ]]; then
    echo "Supply an outcome (misa_pt, multi_class, death)"
    exit 1
fi
if [[ ! "$1" =~ ^(misa_pt|multi_class|death)$ ]]; then
    echo "Outcome must be one of: (misa_pt, multi_class, death)"
    exit 1
fi

# Capture outcome
outcome=$1

# Construct path to expected sequence location
seqs="${PWD}/output/pkl/${outcome}_trimmed_seqs.pkl"

if [[ ! -f "${seqs}" ]]; then
    echo "Expected trimmed sequences for ${outcome} at ${seqs}"
    echo "Have you run model_prep.py?"
    exit 1
fi

# Check if virtualenv present, and if so activate it
venv_path="${PWD}/venv/Scripts/activate"

# Activate virtualenv
if [[ -f "${venv_path}" ]]; then
    echo "Virualenv found, activating"
    source "${venv_path}"
fi

py_location=$(which python)

# Echo the python install and version
echo "${py_location}"


# Run Baselines
run_baseline() {
    python "$PWD/python/baseline_models.py" --outcome=$outcome
}
    
run_dan() {
    # Run DAN
    python "${PWD}/python/model.py" --outcome=$outcome --day_one --model=dan
    python "$PWD/python/model.py" --outcome=$outcome --all_days --model=dan
}

run_lstm() {
    # Run LSTM
    python "$PWD/python/model.py" --outcome=$outcome --all_days --model=lstm
}

# Run Baselines
echo "Running Baselines >>"
run_baseline
echo "Running DAN >>"
run_dan
echo "Running LSTM >>"
run_lstm
