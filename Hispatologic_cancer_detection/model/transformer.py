import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import os
import warnings
from PIL import Image
from Hispatologic_cancer_detection.logs.logs import *
from Hispatologic_cancer_detection.configs.confs import *
from Hispatologic_cancer_detection.transforms.transform import *
from keras import backend as K

def recall_m(y_true, y_pred) -> float:
    """
    The goal of this function is to calculate the recall metric
    as the model is being fitted

    Arguments:
        -y_true: The true labels of the test class
        -y_pred: The predicted labels of the test class

    Returns:
        -recall: float: The computed recall of the model
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred) -> float:
    """
    The goal of this function is to calculate the precision metrics
    as the model is being fitted

    Arguments:
        -y_true: The true labels of the test class
        -y_pred: The predicted labels of the test class

    Returns:
        -precision: float: The computed recall of the model
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred) -> float:
    """
    The goal of this function is to calculate the precision metrics
    as the model is being fitted

    Arguments:
        -y_true: The true labels of the test class
        -y_pred: The predicted labels of the test class

    Returns:
        -precision: float: The computed recall of the model
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main_params = load_conf("configs/main.yml", include=True)

warnings.filterwarnings("ignore")

image_size = main_params["transformer_params"]["image_size"]
batch_size = main_params["transformer_params"]["batch_size"]
n_classes = main_params["transformer_params"]["n_classes"]
validation_split = 1 - main_params["pipeline_params"]["train_size"]

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=validation_split
)
train_data_gen = image_generator.flow_from_directory(
    directory="train",
    subset="training",
    class_mode="binary",
    shuffle=True,
    target_size=(32, 32),
)


val_data_gen = image_generator.flow_from_directory(
    directory="train", subset="validation", class_mode="binary", shuffle=True
)


def data_augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > 0.75:
        image = tf.image.transpose(image)

    # Rotates
    if p_rotate > 0.75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > 0.5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > 0.25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    return image


learning_rate = main_params["transformer_params"]["learning_rate"]
weight_decay = main_params["transformer_params"]["weight_decay"]
num_epochs = main_params["transformer_params"]["num_epochs"]

patch_size = main_params["transformer_params"][
    "patch_size"
]  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = main_params["transformer_params"]["projection_dim"]
num_heads = main_params["transformer_params"]["num_heads"]
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = main_params["transformer_params"]["transformer_layers"]
mlp_head_units = main_params["transformer_params"][
    "mlp_head_units"
]  # Size of the dense layers of the final classifier


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = L.Dense(units, activation=tf.nn.gelu)(x)
        x = L.Dropout(dropout_rate)(x)
    return x


class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(L.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = L.Dense(units=projection_dim)
        self.position_embedding = L.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class Transformer:
    """
    The goal of this class is to stucture all above
    functions in a transformer model
    """

    def __init__(self):
        pass

    def loading_data(self) -> None:

        """
        The goal of this function is loading
        appropriate data for transformer training

        Arguments:
            None

        Returns:
            None
        """
        logging.info("Loading data...")

        df_label = pd.read_csv("train_labels.csv")
        df_train = pd.DataFrame(train_data_gen.filenames, columns=["image_path"])
        df_train["image_name"] = df_train.image_path.apply(
            lambda x: x.replace(".tif", "")
            .replace("0. non_cancerous/", "")
            .replace("1. cancerous/", "")
        )
        
        df_test = pd.DataFrame(val_data_gen.filenames, columns=["image_path"])
        df_test["image_name"] = df_test.image_path.apply(
            lambda x: x.replace(".tif", "")
            .replace("0. non_cancerous/", "")
            .replace("1. cancerous/", "")
        )
        print("df_train !!!!!!", df_train)
        df_train = df_train.merge(df_label, left_on="image_name", right_on="id")
        print("df_label !!!!!!", df_label)

        df_test = df_test.merge(df_label, left_on="image_name", right_on="id")
        df_train = df_train[["image_path", "label"]]
        df_test = df_test[["image_path", "label"]]
        df_train.label, df_test.label = df_train.label.astype(
            str
        ), df_test.label.astype(str)
        
        logging.info("Data Loaded !")

        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=validation_split,
            preprocessing_function=data_augment,
        )

        self.train_gen = self.datagen.flow_from_dataframe(
            dataframe=df_train,
            directory="train",
            x_col="image_path",
            y_col="label",
            subset="training",
            batch_size=batch_size,
            seed=1,
            color_mode="rgb",
            shuffle=True,
            class_mode="categorical",
            target_size=(image_size, image_size),
        )

        self.valid_gen = self.datagen.flow_from_dataframe(
            dataframe=df_train,
            directory="train",
            x_col="image_path",
            y_col="label",
            subset="validation",
            batch_size=batch_size,
            seed=1,
            color_mode="rgb",
            shuffle=True,
            class_mode="categorical",
            target_size=(image_size, image_size),
        )

        self.test_gen = self.datagen.flow_from_dataframe(
            dataframe=df_test,
            x_col="image_path",
            y_col=None,
            batch_size=batch_size,
            seed=1,
            color_mode="rgb",
            shuffle=True,
            class_mode=None,
            target_size=(image_size, image_size),
        )

    def vision_transformer(self):
        inputs = L.Input(shape=(image_size, image_size, 3))

        # Create patches.
        patches = Patches(patch_size)(inputs)

        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):

            # Layer normalization 1.
            x1 = L.LayerNormalization(epsilon=1e-6)(encoded_patches)

            # Create a multi-head attention layer.
            attention_output = L.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)

            # Skip connection 1.
            x2 = L.Add()([attention_output, encoded_patches])

            # Layer normalization 2.
            x3 = L.LayerNormalization(epsilon=1e-6)(x2)

            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = L.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = L.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = L.Flatten()(representation)
        representation = L.Dropout(0.5)(representation)

        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.25)

        # Classify outputs.
        logits = L.Dense(n_classes)(features)
        # Create the model.
        model = tf.keras.Model(inputs=inputs, outputs=logits)

        return model

    def fit(self):
        """
        The goal of this function is launching
        the fitting of the model

        Arguments:
            None

        Returns:
            None
        """
        self.loading_data()

        decay_steps = self.train_gen.n // self.train_gen.batch_size
        initial_learning_rate = learning_rate

        lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate, decay_steps
        )

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = self.vision_transformer()

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                label_smoothing=0.0,
                axis=-1,
                reduction="auto",
                name="binary_crossentropy",
            ),
            metrics=["accuracy", recall_m, precision_m, f1_m],
        )

        STEP_SIZE_TRAIN = self.train_gen.n // self.train_gen.batch_size
        STEP_SIZE_VALID = self.valid_gen.n // self.valid_gen.batch_size

        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=1e-4,
            patience=5,
            mode="max",
            restore_best_weights=False,
            verbose=1,
        )

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath="./model.hdf5",
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )

        callbacks = [earlystopping, lr_scheduler, checkpointer]

        logging.warning("Fitting of the transformer model has begun")

        model.fit(
            x=self.train_gen,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=self.valid_gen,
            validation_steps=STEP_SIZE_VALID,
            epochs=num_epochs,
            callbacks=callbacks,
        )

        logging.warning("Fitting of the transformer model has just finished")
        logging.warning("Saving model...")
        model.save(
            os.path.join(
                os.getcwd(), main_params["transformer_params"]["save_model_path"]
            )
        )
        logging.warning("Model successfuly saved !")

    def predict_label(self, image_path: str, loading_model=True) -> int:
        """
        The goal of this function is, after having received an image,
        to predict the associated label
        Arguments:
            -image_path: str: The path of the
            image which label has to be predicted
        Returns:
            -label: int: The predicted label of the image
        """

        model = self.vision_transformer()

        if loading_model:
            model.load_weights(
                os.path.join(
                    os.getcwd(), main_params["transformer_params"]["save_model_path"]
                )
            )
        else:
            if "train_labels.csv" not in os.listdir():
                liste_image=os.listdir("train/0. non_cancerous")+os.listdir("train/1. cancerous")
                df_train=pd.DataFrame(liste_image,columns=["id"])
                df_train["label"]=1
                df_train.loc[:len(os.listdir("train/0. non_cancerous")),"label"]=0
                df_train.to_csv("train_labels.csv",index=False)

            model.fit()
            logging.info("Model has been fitted for prediction")
        img = Image.open(image_path)
        img = img.resize(
            (
                main_params["pipeline_params"]["resize"],
                main_params["pipeline_params"]["resize"],
            )
        )
        img = np.expand_dims(img, axis=0)

        if loading_model:
            predicted = model.predict_label(img)
            predicted = np.argmax(predicted)
        else:
            predicted = self.model.predict(img)
            predicted = np.argmax(predicted)
        return predicted


if __name__ == "__main__":
    main()
    model = Transformer()
