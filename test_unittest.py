import unittest
import os
import random
import json
from Hispatologic_cancer_detection.visualisation.visualisation import *
from Hispatologic_cancer_detection.transforms.transform import *
from Hispatologic_cancer_detection.configs.confs import *
from Hispatologic_cancer_detection.model.transformer import *
from PIL import Image
from Hispatologic_cancer_detection.model.model import ConvNeuralNet
import flask
import pytest
from Hispatologic_cancer_detection.app.app import *

main_params = load_conf("configs/main.yml", include=True)


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_image_transform(self) -> bool:
        """
        The goal of this test function is to check
        that the transformer function works and returns
        appropriate shape

        Arguments:
            None

        Returns:
            bool: Boolean to check wheter the Transformer 
            function returns tensor with appropriate shape
        """
        transformer = transform()
        img = Image.open(main_params["pipeline_params"]["test_image_path"])
        x = transformer(img)
        self.assertEqual(x.shape[-1], main_params["pipeline_params"]["resize"])
        self.assertEqual(x.shape[-2], main_params["pipeline_params"]["resize"])

    def test_model_pipeline(self) -> bool:
        """
        The goal of this function is to check the model
        pipeline works well without error

        Arguments:
            None

        Returns:
            bool: Boolean to check wheter the function launched
            in the model.py file works from the command line 
        """
        try:
            os.system("python Hispatologic_cancer_detection/model/model.py")
        except:
            self.fail("Error: There has been a problem in training the pipeline")

    def testing_main(self) -> bool:
        """
        The goal of this self function is to check if the
        main.py file works when called

        Arguments:
            None

        Returns:
            None
        """

        try:
            os.system("python main.py Hippolyte")
        except:
            self.fail("Error, failed to achieve the main pipeline ")

    def testing_main_transformer(self) -> bool:
        """
        The goal of this self function is to check if the
        main.py file works when to be used with Transformer
        model

        Arguments:
            None

        Returns:
            bool: Boolean to check wheter the function launched
            in the main.py file works from the command line when 
            calling transformer model
        """

        try:
            os.system("python main.py Hippolyte transformer")
        except:
            self.fail("Error, failed to achieve the main pipeline ")

    def test_prediction_cnn(self) -> bool:
        """
        The goal of this function is to check if the prediction returned by
        the CNN model is coherent, that is that it returns 0 or 1

        Arguments:
            None

        Returns:
            bool: Boolean to check wheter the prediction made by the model 
            makes sense 
        """

        all_images = os.listdir(main_params["non_cancerous_image_path"]) + os.listdir(
            main_params["cancerous_image_path"]
        )
        image_chosen = random.choice(all_images)

        if os.path.exists(
            os.path.join(main_params["cancerous_image_path"], image_chosen)
        ):
            path = os.path.join(main_params["cancerous_image_path"], image_chosen)
        elif os.path.exists(
            os.path.join(main_params["non_cancerous_image_path"], image_chosen)
        ):
            path = os.path.join(main_params["non_cancerous_image_path"], image_chosen)
        model = ConvNeuralNet(main_params["num_classes"])
        prediction = model.predict(image_path=path, loading_model=False)

        coherent_prediction = prediction == 0 or prediction == 1

        self.assertTrue(coherent_prediction)

    def test_prediction_transformer(self) -> bool:
        """
        The goal of this function is to check if the prediction
        returned by the Transformer model is coherent, that is that it
        returns 0 or 1

        Arguments:
            None

        Returns:
            bool: Boolean to check if the prediction made 
            by the algorithm makes sense
        """

        all_images = os.listdir(main_params["non_cancerous_image_path"]) + os.listdir(
            main_params["cancerous_image_path"]
        )
        image_chosen = random.choice(all_images)

        if os.path.exists(
            os.path.join(main_params["cancerous_image_path"], image_chosen)
        ):
            path = os.path.join(main_params["cancerous_image_path"], image_chosen)
        elif os.path.exists(
            os.path.join(main_params["non_cancerous_image_path"], image_chosen)
        ):
            path = os.path.join(main_params["non_cancerous_image_path"], image_chosen)
        model = Transformer()
        prediction = model.predict_label(image_path=path, loading_model=False)

        coherent_prediction = prediction == 0 or prediction == 1

        self.assertTrue(coherent_prediction)

    def test_base_route(self)->bool:
        """
        The goal of this first function is to 
        check whether the application is correctly
        launched
        
        Arguments:
            None
        
        Returns:
            bool: Boolean to check if the welcome page
            was reached
        """
        client = app.test_client()
        url = '/'

        response = client.get(url)
        html = response.data.decode()
        assert "Hispatologic Cancer Detection application" in html
        assert response.status_code == 200

    def test_cnn_model(self)->bool:
        """
        The goal of this function is to check wheter 
        the cnn model receives the good parameters
        
        Arguments:
            None 
        Returns:
            bool: Boolean to check if the good parameters
            were received
        """
        client = app.test_client()
        url = '/training_cnn'
        sent={'epochs':3,
        'train_size':0.8,
        'lr':10e-5,
        'batch_size':300,
        'dropout':0.001,
        'weight_decay':0.001}

        result = client.post(
                url,
                data=sent
            )

        self.assertEqual(
                result.data,
                json.dumps(sent)
            )

if __name__ == "__main__":
    unittest.main()
