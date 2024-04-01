# %%
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten,MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from keras.layers import Dropout, InputLayer, Flatten
import keras
import matplotlib.pyplot as plt
from keras.layers import Dropout, InputLayer, Flatten
from keras import regularizers

# %%
def cnn1():

    training_set = preprocessing.image_dataset_from_directory(r"sea_animals", validation_split=0.2,subset="training",label_mode="categorical",seed=0,
    image_size=(100,100))

    test_set = preprocessing.image_dataset_from_directory(r"sea_animals",validation_split=0.2,subset="validation",label_mode="categorical",seed=0,
    image_size=(100,100))

    print("Classes:", training_set.class_names)

    # build the model
    m = Sequential()
    m.add(Rescaling(1/255))
    m.add(Conv2D(32, kernel_size=(3, 3),
    activation='relu',
    input_shape=(100,100,3)))

    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(5, activation='softmax'))
    # setting and training
    m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
    epochs = 25
    print("Training.")
    for i in range(epochs) :

        history = m.fit(training_set, batch_size=32, epochs=1,verbose=0)
        print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])
    # testing
    print("Testing.")
    score = m.evaluate(test_set, verbose=0)
    print('Test accuracy:', score[1])
    # saving the model
    print("Saving the model in my_cnn.h5.")
    m.save("my_cnn.h5")

cnn1()

# %%
def cnn_2():


     
    training_set = preprocessing.image_dataset_from_directory(r"sea_animals", validation_split=0.2,subset="training",label_mode="categorical",seed=0,
    image_size=(100,100))

    test_set = preprocessing.image_dataset_from_directory(r"sea_animals",validation_split=0.2,subset="validation",label_mode="categorical",seed=0,
    image_size=(100,100))

    print("Classes:", training_set.class_names)

    # build the model
    m = Sequential()
    m.add(Rescaling(1/255))
    m.add(Conv2D(32, kernel_size=(3, 3),
    activation='relu',
    input_shape=(100,100,3)))

    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(5, activation='softmax'))
  

    # setting and training
    m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
    epochs = 45
    print("Training.")
    for i in range(epochs) :


        history = m.fit(training_set, batch_size=25, epochs=1,verbose=0)
        print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])
    # testing
    print("Testing.")
    score = m.evaluate(test_set, verbose=0)
    print('Test accuracy:', score[1])
    # saving the model
    print("Saving the model in my_cnn1.h5.")
    m.save("my_cnn1.h5")
cnn_2()

# %%
def fine_tune():

    from tensorflow.keras.applications import VGG16
    """ Trains and evaluates CNN image classifier on the sea animals dataset.
    Saves the trained model. """
    # load datasets
    training_set = preprocessing.image_dataset_from_directory(r"sea_animals",
    validation_split=0.2,
    subset="training",
    label_mode="categorical",
    seed=0,
    image_size=(150,150))
    test_set = preprocessing.image_dataset_from_directory(r"sea_animals",
    validation_split=0.2,
    subset="validation",
    label_mode="categorical",
    seed=0,
    image_size=(150,150))
    print("Classes:", training_set.class_names)
    # Load a general pre-trained model.
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output # output layer of the base model
    x = GlobalAveragePooling2D()(x)
    # a fully-connected layer
    x = Dense(1064, activation='relu')(x)
    output_layer = Dense(5, activation='softmax')(x)
    # this is the model we will train
    m1 = Model(inputs=base_model.input, outputs=output_layer)
    # train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional base model layers
    for layer in base_model.layers:

        layer.trainable = False
    m1.compile(loss="categorical_crossentropy", metrics=['accuracy'])
    epochs = 15
    print("Training.")
    for i in range(epochs) :

        history = m1.fit(training_set, batch_size=64, epochs=1,verbose=0)
        print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])
    #history = m.fit(training_set, batch_size=32, epochs=5,verbose=1)
    print(history.history["accuracy"])
    # testing
    print("Testing.")
    score = m1.evaluate(test_set, verbose=0)
    print('Test accuracy:', score[1])
    # saving the model
    print("Saving the model in my_fine_tuned.h5.")
    m1.save("my_fine_tuned.h5")
fine_tune()

# %%
from tensorflow.keras.models import load_model
import os
# load the image
def test_image(m1,img_file):

    
    img = preprocessing.image.load_img(img_file,target_size=(200,200))
    img_arr = preprocessing.image.img_to_array(img)
    # show the image
    plt.imshow(img_arr.astype("uint8"))
    plt.show()
    # classify the image
    img_cl = img_arr.reshape(1,200,200,3)
    score = m1.predict(img_cl)
    print(score.round(3))

image_folder = r"./10 images"
m1=load_model(r"./my_fine_tuned.h5")
# Loop over the files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
         img_file = os.path.join(image_folder, filename)
         test_image(m1,img_file)

# %%
from tensorflow.keras.models import load_model
import os
# load the image
def test_image2(m1,img_file):

    
    img = preprocessing.image.load_img(img_file,target_size=(100,100))
    img_arr = preprocessing.image.img_to_array(img)
    # show the image
    plt.imshow(img_arr.astype("uint8"))
    plt.show()
    # classify the image
    img_cl = img_arr.reshape(1,100,100,3)
    score = m1.predict(img_cl)
    print(score.round(3))

image_folder = r"./10 images"
m1=load_model(r"./my_cnn.h5")
# Loop over the files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
         img_file = os.path.join(image_folder, filename)
         test_image2(m1,img_file)

# %%



