from PIL import Image,ImageOps
from model.model import *
from tensorflow.keras.models import load_model
import os
from helper.helper import *

mean = 0
std_deviation = 0

if os.path.isfile("savedModel"):
    model = load_model("savedModel")
else:
    train, test = load_data()
    X_train, Y_train = split_and_transform(train)
    X_test, Y_test = split_and_transform(test)
    model = create_model(X_train, Y_train, X_test, Y_test)
    model.save('savedModel', save_format="h5")

filename = "test-images/test3.jpg"
img = Image.open(filename)
img = ImageOps.grayscale(img)
img = img.resize((28,28))
category = model.predict_classes(img)
print(category)
for i in category:
    print(i)
