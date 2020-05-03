import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove debugging logs
import tensorflow as tf
import numpy as np

from skimage.transform import resize
from PIL import Image as PImage

# return dict of images
def loadImages(path: str):
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    imagesList = os.listdir(path)
    loadedImages = {}
    for image in imagesList:
        img = PImage.open(os.path.join(path, image))
        resized_img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        loadedImages[image] = resized_img

    return loadedImages
print("CAT - DOG Classifier")
#print("create a folder with your images called imgs")
path = os.path.join(os.getcwd(), "imgs")

# load model 
print("Loading model...")
model = tf.keras.models.load_model(os.path.join("models", "cat_dog_model"))

# your images in a dict
labels = ["cat", "dog"] # 0 = cat, 1 = dog
imgs = loadImages(path)
for img_name in imgs:
    # you can show every image
    probabilities = model.predict(np.array([np.asarray(imgs[img_name])]))
    print(probabilities)
    print(probabilities[0,:])
    index = np.argsort(probabilities[0,:])
    print(img_name, "is a", labels[index[1]])


