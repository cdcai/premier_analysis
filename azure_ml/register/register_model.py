import argparse
import os
from azureml.core import Run, Dataset
import tensorflow as tf
import sklearn
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
    parser.add_argument("--demog_dict_file",type=str)
    parser.add_argument("--all_ftrs_dict_file",type=str)

    args = parser.parse_args()
    MOD_NAME = args.model
    if 'lstm' not in MOD_NAME:
        MOD_NAME +='_d1'
    OUTCOME = args.outcome

    run = Run.get_context()
    ws = run.experiment.workspace

    model_path = os.path.join(args.model_file, MOD_NAME + "_" + OUTCOME + ".h5")
    vocab_path = os.path.join(args.all_ftrs_dict_file, "all_ftrs_dict.pkl")
    vocab_demog_path = os.path.join(args.demog_dict_file, "demog_dict.pkl")

    print("model path:",model_path)
    print("vocab path:",vocab_path)
    print("vocab demog path:",vocab_demog_path)


    # register the model
    AMLModel.register(workspace=ws,
                      model_name=f"{MOD_NAME}_{OUTCOME}",
                      model_path=model_path,
                      model_framework=AMLModel.Framework.TENSORFLOW,
                      model_framework_version=tf.__version__,
                      resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                      description=f"Classification model to predict {OUTCOME} using {MOD_NAME}",
                       tags={'area': 'EHR', 'type': 'classification'})

    # register the all_ftrs_dict_file vocabulary
    AMLModel.register(workspace=ws,
                      model_name=f"vocab_{MOD_NAME}_{OUTCOME}",
                      model_path=vocab_path,
                      model_framework=AMLModel.Framework.SCIKITLEARN,
                      model_framework_version=sklearn.__version__,
                      resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                      description=f"Vocabulary Temporal representation of EHR Classification model to predict {OUTCOME} using {MOD_NAME}",
                       tags={'area': 'EHR', 'type': 'Vocab'})

    # register the demog_dict_file vocabulary for demographic data
    AMLModel.register(workspace=ws,
                      model_name=f"vocab_demog_{MOD_NAME}_{OUTCOME}",
                      model_path=vocab_demog_path,
                      model_framework=AMLModel.Framework.SCIKITLEARN,
                      model_framework_version=sklearn.__version__,
                      resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                      description=f"Demographic patient Vocabulary Temporal representation of EHR Classification model to predict {OUTCOME} using {MOD_NAME}",
                       tags={'area': 'EHR', 'type': 'Vocab'})

if __name__ == '__main__':
    main()