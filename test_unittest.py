import unittest
from src.visualisation.visualisation import *

class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_image_opening(self)->None:
        """
        The goal of this test function is to check 
        if the function opening images work
        
        Arguments:
            None
        
        Returns:
            None
        """

        try:
            image_visualisation(keep=False)
        except:
            self.fail("Error")

    
if __name__ == "__main__":
    unittest.main()