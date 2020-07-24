from random import randint
import numpy as np
import cv2
import tensorflow as tf

# SETTINGS
IMG_SHAPE = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE


def process_path(file_path):
    # load the raw data from the file as a string
    raw_data = tf.io.read_file(file_path)
    # decode
    return tf.image.decode_jpeg(raw_data, channels=3)


def normalize(image):
    image = tf.cast(image, tf.float32)
    return image / 255


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image,
                            size=[286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = tf.image.random_crop(image, size=[IMG_SHAPE, IMG_SHAPE, 3])

    # random mirroring
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_up_down(image)

    return image


@tf.function
def preprocess_image_train(file_path):
    image = process_path(file_path)
    image = random_jitter(image)
    image = normalize(image)
    return image


@tf.function
def preprocess_image_test(file_path):
    image = process_path(file_path)
    image = tf.image.resize(image, size=[IMG_SHAPE, IMG_SHAPE])
    image = normalize(image)
    return image


def random_mask(image_shape=IMG_SHAPE, channels=3):
    img = np.ones((image_shape, image_shape, channels), dtype='float32')

    # Set size scale
    size = int((image_shape + image_shape) * 0.03)
    if image_shape < 64 or image_shape < 64:
        raise Exception("Image shape of mask must be at least 64!")

    # Draw random lines
    for _ in range(randint(1, 10)):
        x1, x2 = randint(1, image_shape), randint(1, image_shape)
        y1, y2 = randint(1, image_shape), randint(1, image_shape)
        thickness = randint(3, size)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)

    # Draw random circles
    for _ in range(randint(1, 10)):
        x1, y1 = randint(1, image_shape), randint(1, image_shape)
        radius = randint(3, size)
        cv2.circle(img, (x1, y1), radius, (0, 0, 0), -1)

    # Draw random ellipses
    for _ in range(randint(1, 10)):
        x1, y1 = randint(1, image_shape), randint(1, image_shape)
        s1, s2 = randint(1, image_shape), randint(1, image_shape)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (0, 0, 0), thickness)

    return img


def create_input_pipeline(dir, batch_size, buffer_size=1000):
    list_ds = tf.data.Dataset.list_files(dir + '/**/*.jpg')
    dataset = list_ds.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size).repeat()
    return dataset.prefetch(buffer_size=AUTOTUNE)


def create_input_pipeline_test(dir, batch_size, buffer_size=15):
    list_ds = tf.data.Dataset.list_files(dir + '/*.jpg')
    dataset = list_ds.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size)


def create_input_dataset(image_batch, batch_size):
    mask_batch = np.stack(
        [random_mask(image_shape=IMG_SHAPE) for _ in range(batch_size)],
        axis=0)
    # Condition tensor
    bool_mask_batch = tf.convert_to_tensor(mask_batch.copy().astype('bool'),
                                           dtype=tf.bool)

    mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)
    masked_batch = tf.where(bool_mask_batch, image_batch, 1.0)

    return (masked_batch, mask_batch), image_batch
