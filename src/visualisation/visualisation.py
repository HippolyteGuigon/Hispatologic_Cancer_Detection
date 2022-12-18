import cv2
import os
import random
import numpy as np


def image_visualisation(
    image="f0c2a0b8ef3024f407fa97d852d49be0215cafe0.tif", flag=-1, keep=True
) -> None:
    """
    The goal of this function is to display a given image
    for a certain time until the user hits a command

    Arguments:
        -image: str: The name of the image the user wants to
        display
        -flag: int : The way the image is loaded, how colours
        appear

    Returns:
        None
    """

    if os.path.exists(os.path.join("train/cancerous", image)):
        path = os.path.join("train/cancerous", image)
    elif os.path.exists(os.path.join("train/non_cancerous", image)):
        path = os.path.join("train/non_cancerous", image)
    else:
        path = os.path.join("test", image)

    img = cv2.imread(path, flag)
    imS = cv2.resize(img, (400, 200))
    cv2.imshow("image", imS)
    if keep:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def class_visualisation() -> None:
    """
    The goal of this function is to display multiple images
    different classes to see what are the characteristics of
    cancerous and non-cancerous images

    Arguments:
        None

    Returns:
        None
    """

    cancerous_images_list = os.listdir("train/cancerous")
    non_cancerous_images_list = os.listdir("train/non_cancerous")
    cancerous_image_choice = random.choices(cancerous_images_list, k=4)
    non_cancerous_image_choice = random.choices(non_cancerous_images_list, k=4)

    img1_canc = cv2.imread(os.path.join("train/cancerous", cancerous_image_choice[0]))
    img2_canc = cv2.imread(os.path.join("train/cancerous", cancerous_image_choice[1]))
    img3_canc = cv2.imread(os.path.join("train/cancerous", cancerous_image_choice[2]))
    img4_canc = cv2.imread(os.path.join("train/cancerous", cancerous_image_choice[3]))

    img_1_ncanc = cv2.imread(
        os.path.join("train/non_cancerous", non_cancerous_image_choice[0])
    )
    img_2_ncanc = cv2.imread(
        os.path.join("train/non_cancerous", non_cancerous_image_choice[1])
    )
    img_3_ncanc = cv2.imread(
        os.path.join("train/non_cancerous", non_cancerous_image_choice[2])
    )
    img_4_ncanc = cv2.imread(
        os.path.join("train/non_cancerous", non_cancerous_image_choice[3])
    )

    # concatenate image Horizontally
    cancerous_stack = np.concatenate(
        (img1_canc, img2_canc, img3_canc, img4_canc), axis=1
    )
    non_cancerous_stack = np.concatenate(
        (img_1_ncanc, img_2_ncanc, img_3_ncanc, img_4_ncanc), axis=1
    )

    cv2.imshow("CANCEROUS", cancerous_stack)
    cv2.imshow("NON CANCEROUS", non_cancerous_stack)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


class_visualisation()
