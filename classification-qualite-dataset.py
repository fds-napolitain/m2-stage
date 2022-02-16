import tensorflow as tf
import cv2

# path
PATH = "datasets/Qualite"

# étiquettes
class_names = ["FaceQual0","FaceQual1","FaceQual2","FaceQual3"]

# load dataset
batch_size = 32
img_height = 300
img_width = 300
IMG_SIZE = (img_width, img_height)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="int"
)
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    PATH,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

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

# Crée jeu de tests
val_batches = tf.data.experimental.cardinality(validation_ds)
test_dataset = validation_ds.take(val_batches // 3)
validation_ds = validation_ds.skip(val_batches // 3)
