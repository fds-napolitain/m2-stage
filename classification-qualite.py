import cv2
import tensorflow as tf
import numpy as np
import pandas


def _downgrade(img, blur: int, downscale: int, kernel):
    """
    Downgrade les images FaceQual3 en images FaceQual0
    :param img: image
    :param blur: niveau de flou
    :param downscale: niveau de downscale upscale (pixelisation)
    :param kernel: kernel du filtre de floutage gaussien
    """
    img = tf.image.resize(img, [downscale, downscale], method="nearest")
    img = tf.image.resize(img, [300, 300], method="nearest")
    output = cv2.filter2D(img.numpy(), -1, kernel)
    return output


class CustomDataset:
    """
    Dataset pour le modèle
    """

    def __init__(self, path: str, class_names=None, img_width=300, img_height=300, split=0.3):
        self.validation_ds = None
        self.test_dataset = None
        self.val_batches = None
        self.train_ds = None
        self.path = path
        self.class_names = class_names
        self.batch_size = 32
        self.img_width = img_width
        self.img_height = img_height
        self.img_size = (self.img_width, self.img_height)
        self.img_shape = (self.img_width, self.img_height) + (3,)
        self.split = split
        self.img_shape = (img_width, img_height) + (3,)

    def reload(self, batch_size=32):
        """
        Charge / recharge le dataset à partir du path dans le constructeur vers celui ci.
        :param batch_size:
        :return:
        """
        self.batch_size = batch_size
        # Crée jeu d'entrainement
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path,
            validation_split=0.3,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            label_mode="int"
        )
        # Crée jeu de validation
        self.validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path,
            validation_split=0.3,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size
        )
        # Crée jeu de tests, peut etre pas mettre l'augmentation dans le test! (risque de résultat biaisé)
        self.val_batches = tf.data.experimental.cardinality(self.validation_ds)
        self.test_dataset = self.validation_ds.take(self.val_batches // 3)
        self.validation_ds = self.validation_ds.skip(self.val_batches // 3)

    def downgrade(self, img, blur=9, downscale=40):
        """
        Downgrade les images FaceQual3 en images FaceQual0
        :param img: image
        :param blur: niveau de flou
        :param downscale: niveau de downscale upscale (pixelisation)
        """
        kernel = np.ones((blur, blur), np.uint8) / pow(blur, 2)
        nb = 1260 - 254  # a fare plus propre / nombre de photos a generer pour FaceQual0
        for x, y in self.train_ds:
            for i in range(len(y.numpy())):
                if y[i].numpy() == 3 and nb > 0:
                    cv2.imwrite("datasets/Qualite/FaceQual0/generated_" + str(nb) + ".jpg", cv2.cvtColor(_downgrade(x[i], blur, downscale, kernel).numpy(), cv2.COLOR_RGB2BGR))
                    nb -= 1


class CustomModel:
    """
    Modèle pour la qualité des photos.
    """

    def __init__(self, dataset: CustomDataset):  # a splitter apres pour rajouter les layers a la vole
        self.history = None
        self.accuracy0 = None
        self.loss0 = None
        self.dataset = dataset
        inputs = tf.keras.Input(shape=(dataset.img_width, dataset.img_height, 3))
        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)(inputs)
        x = tf.keras.applications.MobileNetV2(input_shape=self.dataset.img_shape,
                                               include_top=False,
                                               weights='imagenet')(x, training=False)  # mobile net v2
        x = tf.keras.layers.GlobalAveragePooling2D(x)  # moyenne des features
        x = tf.keras.layers.Dense(1280, activation="relu")(x)  # convolution 2d
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1280, activation="relu")(x)  # convolution 2d
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1280, activation="relu")(x)  # convolution 2d
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)  # prediction
        self.model = tf.keras.Model(inputs, outputs)

    def compile(self, learning_rate=0.0001):
        """
        Compile modèle
        :param learning_rate: learning rate, 0.0001 par défaut
        """
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=[
                               "accuracy",
                           ])

    def run(self, epochs=10):
        """
        Lance un entrainement
        :param epochs: nombre d'epochs, 10 par défaut
        """
        self.loss0, self.accuracy0 = self.model.evaluate(self.dataset.validation_ds)
        self.history = self.model.fit(self.dataset,
                                      epochs=epochs,
                                      validation_data=self.dataset.validation_ds)

    def save_history(self, filename: str):
        """
        Enregistre un modèle + l'historique de l'entrainement
        :param filename: nom du modèle/historique
        """
        self.model.save("saved_models/" + filename + ".h5")
        pandas.DataFrame(self.history.history).to_csv("saved_model/" + filename + ".csv")

    def load_history(self, filename: str):
        """
        Charge un modèle + l'historique de l'entrainement
        :param filename: nom du modèle/historique
        """
        self.model = tf.keras.models.load_model("saved_models/" + filename + ".h5")
        self.history = pandas.read_csv(filename)
