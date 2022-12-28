from src.logs.logs import *
from src.configs.confs import *
import argparse
from src.model.model import *
from src.model_save_load.model_save_load import *
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
    df_test = pd.read_csv("sample_submission.csv")
    df_test["id"] = df_test["id"].apply(lambda x: x + str(".tif"))
    predictor = ConvNeuralNet(2)
    df_test["label"] = df_test["id"].progress_apply(
        lambda image: predictor.predict("test/" + image).item()
    )
    df_test["id"] = df_test["id"].apply(lambda x: x.replace(".tif", ""))
    df_test.to_csv("sample_submission.csv", index=False)


if __name__ == "__main__":
    main()
    if args.Model == "model_train":
        launch_pipeline()
    model = load_model()
    if args.Test_predict == "predict":
        predict_test_file()
