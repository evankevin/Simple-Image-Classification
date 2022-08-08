import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np


def ImageProcess(filepath, *, color=None):
    width = 200
    height = 200
    image = cv2.imread(filepath)
    if color != None:
        image = cv2.cvtColor(image, color)
    imageRescale = cv2.resize(image, (width, height), cv2.INTER_AREA)
    imageRescale = imageRescale.astype(np.float64)
    imageRescale /= 255
    return imageRescale


def GetCategory():
    Trainmapping = {}
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), "Training Model"))):
        Trainmapping[number] = folder
    return Trainmapping


def GetFiles():
    imgTrainList = []
    imgName = []
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), "Training Model"))):
        for image_name in os.listdir(os.path.join(os.getcwd(), "Training Model", folder)):
            imgName.append(number)
            image = ImageProcess(os.path.join(os.getcwd(
            ), "Training Model", folder, image_name))
            imgTrainList.append(image)
    return (imgTrainList, imgName)


def CreateModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


def Train(model, train_image, train_filename):
    checkpoint_path = "WeightCheckpoint/model.ckpt"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(train_image, train_filename, epochs=15,
              callbacks=[checkpoint_callback])


def Predict(model, mapping):
    images = []
    results = []
    images_RGB = []
    for images_path in os.listdir(os.path.join(os.getcwd(), "Image Input")):

        image = ImageProcess(os.path.join(
            os.getcwd(), "Image Input", images_path))
        images.append(image)
        image = np.array(image)
        result = model.predict(np.expand_dims(image, axis=0))
        result = np.argmax(result[0])
        results.append(mapping[result])
        imageRGB = ImageProcess(os.path.join(os.getcwd(
        ), "Image Input", images_path), color=cv2.COLOR_BGR2RGB)
        images_RGB.append(imageRGB)

    plt.figure(figsize=(9, 6))
    for i in range(len(images_RGB)):
        fig = plt.gcf()
        fig.canvas.set_window_title('Image Classification')
        plt.subplot(3, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(results[i])
        plt.imshow(images_RGB[i])
    plt.show()


train_image, train_filename = GetFiles()
train_image = np.array(train_image)
train_filename = np.array(train_filename)

model = CreateModel()

Train(model, train_image, train_filename)
model.save('TrainedModel')

model.load_weights(os.path.join(os.getcwd(), 'WeightCheckpoint', 'model.ckpt'))
model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'TrainedModel'))
Predict(model, GetCategory())
