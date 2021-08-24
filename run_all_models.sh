#!/usr/bin/env bash
# == Automate running all models for every outcome

# Define outcomes
declare -a Outcomes=("death" "misa_pt" "multi_class")

# Check if virtualenv present, and if so activate it
venv_path="${PWD}/venv/Scripts/activate"

# Activate virtualenv
if [[ -f "${venv_path}" ]]; then
    echo "Virualenv found, activating"
    source "${venv_path}"
fi

py_location=$(where python || which python)

# Construct path to expected sequence location
seqs="${PWD}/output/pkl/${outcome}_trimmed_seqs.pkl"

# Helper functions
run_model_prep() {
    # TODO: We could also handle other options here
    python "$PWD/python/model_prep.py" --outcome=$1
}

run_models() {
    source "${PWD}/run_models.sh" $1
}

compute_cis() {
    # Compute Bootstrapped CIs
    python "$PWD/python/analysis.py" --parallel
}

# Run models for every outcome
for outcome in ${Outcomes[*]}; do
    echo ">> Running models for $outcome"
    run_models $outcome
done

echo "Computing Bootstrapped CIs"
compute_cis
echo "Done!"
