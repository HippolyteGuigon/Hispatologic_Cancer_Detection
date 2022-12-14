import torch
import cv2
import os


def image_visualisation(image="f0c2a0b8ef3024f407fa97d852d49be0215cafe0.tif") -> None:
    """
    The goal of this function is to display a given image
    for a certain time until the user hits a command

    Arguments:
        -image: The name of the image the user wants to
        display

    Returns:
        None
    """

    if os.path.exists(os.path.join("train", image)):
        path = os.path.join("train", image)
    else:
        path = os.path.join("test", image)

    img = cv2.imread(path)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
