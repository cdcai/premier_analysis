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

# Check if virtualenv present, and if so activate it
venv_path="${PWD}/venv/Scripts/activate"

# Activate virtualenv
if [[ -f "${venv_path}" ]]; then
    echo "Virualenv found, activating"
    source "${venv_path}"
fi

py_location=$(where python || which python)

# TODO: possibly delete existing preds and coefs to avoid overwriting?

# Echo the python install and version
echo "${py_location}"

run_model_prep() {
    # TODO: We could also handle other options here
    python "$PWD/python/model_prep.py" --outcome=$outcome
}

# Run Baselines
run_baseline() {
    echo "-- Day one model --"
    python "$PWD/python/baseline_models.py" --day_one --outcome=$outcome
    echo "-- All day model --"
    python "$PWD/python/baseline_models.py" --all_days --outcome=$outcome
}
    
run_dan() {
    # Run DAN
    echo "-- Day one model --"
    python "${PWD}/python/model.py" --outcome=$outcome --day_one --model=dan
    echo "-- All day model --"
    python "$PWD/python/model.py" --outcome=$outcome --all_days --model=dan
}

run_lstm() {
    # Run LSTM
    python "$PWD/python/model.py" --outcome=$outcome --all_days --model=lstm
}

compute_cis() {
    # Compute Bootstrapped CIs
    python "$PWD/python/analysis.py" --outcome=$outcome --day_one
}

echo "Trimming sequences and appending labels >>"
run_model_prep
echo "Running Baselines >>"
run_baseline
echo "Running DAN >>"
run_dan
echo "Running LSTM >>"
run_lstm
echo "Computing Bootstrap CIs >>"
compute_cis