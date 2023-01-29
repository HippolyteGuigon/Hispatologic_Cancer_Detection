from Hispatologic_cancer_detection.logs.logs import *
from Hispatologic_cancer_detection.configs.confs import *
import argparse
from Hispatologic_cancer_detection.model.model import *
from Hispatologic_cancer_detection.model.cnn import *
from Hispatologic_cancer_detection.model_save_load.model_save_load import *
from tqdm import tqdm

tqdm.pandas()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main_params = load_conf("configs/main.yml", include=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "Name",
    help="The name entered by the user to easily find your own iteration in the logs.",
    nargs="?",
    const="Hippolyte",
    type=str,
)

parser.add_argument(
    "Model",
    help="The provenance of the model that will be used. If you\
        want the model to be trained, enter model_train, else enter\
        get_model",
    nargs="?",
    const="get_model",
    type=str,
)

parser.add_argument(
    "Test_predict",
    help="Command to determine if the test file for the\
        Kaggle competition should be predicted",
    nargs="?",
    const="get_model",
    type=str,
)

args = parser.parse_args()


def launch_pipeline() -> None:
    """
    The goal of this function is to launch the model
    global pipeline with the logs accorded to the commands
    entered by the user

    Arguments:
        None

    Returns:
        None
    """
    chosen_model = main_params["model_chosen"]
    logging.info(f"You launched the model iteration {args.Name}")
    logging.info(f"You have chosen the model {chosen_model} {args.Name}")
    launch_model()
    logging.warning(f"The model has finished training {args.Name}")


def predict_test_file() -> None:
    """
    The goal of this function is to fulfill the test
    file that will be used for Kaggle competition

    Arguments:
        None

    Returns:
        None
    """
    chosen_model = main_params["model_chosen"]
    df_test = pd.read_csv("sample_submission.csv")
    df_test["id"] = df_test["id"].apply(lambda x: x + str(".tif"))
    if chosen_model == "cnn":
        predictor = ConvNeuralNet(main_params["num_classes"])
        df_test["label"] = df_test["id"].progress_apply(
            lambda image: predictor.predict("test/" + image)
        )

    elif chosen_model == "transformer":
        predictor = Transformer.vision_transformer()
        predictor.load_weights(
            os.path.join(
                os.getcwd(), main_params["transformer_params"]["save_model_path"]
            )
        )

        df_test["label"] = df_test["id"].progress_apply(
            lambda image: predictor.predict_label("test/" + image)
        )

    df_test["id"] = df_test["id"].apply(lambda x: x.replace(".tif", ""))
    df_test.to_csv("sample_submission.csv", index=False)


if __name__ == "__main__":
    main()
    chosen_model = main_params["model_chosen"]
    if args.Model == "model_train":
        launch_pipeline()
    if chosen_model == "cnn":
        model = load_model()
    elif chosen_model == "transformer":
        predictor = Transformer.vision_transformer()
        predictor.load_weights(
            os.path.join(
                os.getcwd(), main_params["transformer_params"]["save_model_path"]
            )
        )
    if args.Test_predict == "predict":
        predict_test_file()
