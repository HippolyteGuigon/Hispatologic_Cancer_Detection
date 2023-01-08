import unittest
import os
import random
from Hispatologic_cancer_detection.visualisation.visualisation import *
from Hispatologic_cancer_detection.transforms.transform import *
from Hispatologic_cancer_detection.configs.confs import *
from Hispatologic_cancer_detection.model.model import ConvNeuralNet
from PIL import Image

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
            None
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
            None
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

    def test_prediction(self)->int:
        """
        The goal of this function is to check if the 
        prediction returned by the model is coherent,
        that is that it returns 0 or 1

        Arguments:
            None
        
        Returns:
            None
        """

        all_images=os.listdir(main_params["non_cancerous_image_path"])+os.listdir(main_params["cancerous_image_path"])
        image_chosen=random.choice(all_images)

        if os.path.exists(os.path.join(main_params["cancerous_image_path"], image_chosen)):
            path = os.path.join(main_params["cancerous_image_path"], image_chosen)
        elif os.path.exists(os.path.join(main_params["non_cancerous_image_path"], image_chosen)):
            path = os.path.join(main_params["non_cancerous_image_path"], image_chosen)
        model=ConvNeuralNet(main_params["num_classes"])
        prediction=model.predict(image_path=path,load_model=False)

        if prediction==0 or prediction==1:
            message_predict="Succes"
        else:
            message_predict="Failed"
        self.assertEqual(message_predict,"Succes") 

if __name__ == "__main__":
    unittest.main()
