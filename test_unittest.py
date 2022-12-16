import unittest
from src.visualisation.visualisation import *
from src.transforms.transform import *
from src.configs.confs import *
from PIL import Image
main_params = load_conf("configs/main.yml", include=True)

class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_image_transform(self)->None:
        """
        The goal of this test function is to check 
        that the transformer function works and returns 
        appropriate shape
        
        Arguments:
            None
        
        Returns:
            None
        """
        transformer=transform()
        img = Image.open(main_params["pipeline_params"]["test_image_path"])
        x = transformer(img)
        self.assertEqual(x.shape[-1],main_params["pipeline_params"]["resize"])
        self.assertEqual(x.shape[-2],main_params["pipeline_params"]["resize"])

    
if __name__ == "__main__":
    unittest.main()