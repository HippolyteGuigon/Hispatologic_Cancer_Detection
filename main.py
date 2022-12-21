from src.logs.logs import *
from src.configs.confs import *
import argparse
from src.model.model import *

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

args = parser.parse_args()


def launch_pipeline() -> None:
    chosen_model = main_params["model_chosen"]
    logging.info(f"You launched the model iteration {args.Name}")
    logging.info(f"You have chosen the model {chosen_model} {args.Name}")
    launch_model()
    logging.warning(f"The model has finished training {args.Name}")


if __name__ == "__main__":
    main()
    launch_pipeline()
