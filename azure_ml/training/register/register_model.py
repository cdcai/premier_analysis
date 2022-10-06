import argparse
import os
from azureml.core import Run, Dataset
import tensorflow as tf
from azureml.core.model import Model as AMLModel
from azureml.core.resource_configuration import ResourceConfiguration

def main():


    parser = argparse.ArgumentParser("register")
    
    parser.add_argument("--model_file", type=str, help="model file")
    parser.add_argument("--model",
                        type=str,
                        default="dan",
                        choices=["dan", "lstm", "hp_lstm", "hp_dan"],
                        help="Type of Keras model to use")
    parser.add_argument("--outcome",
                        type=str,
                        default="misa_pt",
                        choices=["misa_pt", "multi_class", "death", "icu"],
                        help="which outcome to use as the prediction target")
    
    args = parser.parse_args()
    MOD_NAME = args.model
    OUTCOME = args.outcome
    
    run = Run.get_context()
    ws = run.experiment.workspace

    model_path = os.path.join(args.model_file, MOD_NAME + "_d1" + "_" + OUTCOME + ".h5")

    print("model path:",model_path)

    AMLModel.register(workspace=ws,
                      model_name=f"{MOD_NAME}_{OUTCOME}",
                      model_path=model_path,
                      model_framework=AMLModel.Framework.TENSORFLOW,
                      model_framework_version=tf.__version__,
                      resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                      description=f"Classification model to predict {OUTCOME} using {MOD_NAME}",
                       tags={'area': 'EHR', 'type': 'classification'})

if __name__ == '__main__':
    main()