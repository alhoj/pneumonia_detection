import pneumonia_detection
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import set_random_seed
from keras.callbacks import EarlyStopping

# set random seed for reproducibility
set_random_seed(1234)

## Preprocess data
# dimensions to which all images will be resized 
target_size = (64, 64)
# use various parameters to augment training data
print('Training data:')
train_set = pneumonia_detection.preprocessing('train', target_size, rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, 
                                            rotation_range=40, width_shift_range=0.2, height_shift_range=0.2)
print('Validation data:')
val_set = pneumonia_detection.preprocessing('val', target_size, rescale=1/255)
print('Testing data:')
test_set = pneumonia_detection.preprocessing('test', target_size, rescale=1/255)

## Build the model
# convolutional neural networks with linear stacks of layers (i.e., no feedback connections)
model = Sequential()
# add convolutional layer with 32 filters of size 3x3, rectified linear unit (ReLU) as activation function,
model.add(Conv2D(32, (3,3), activation='relu', input_shape=target_size+(1,), padding='same'))
# add pooling layer of size 2x2 (downsampling) to reduce computational complexity
model.add(MaxPooling2D(pool_size=(2,2)))
# add another convolutional layer
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
# and another pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# and another convolutional layer with more filters
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
# and another pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# flatten the result (multidimensional vector to one-dimensional vector); needed between convolution/pooling layers and dense layer
model.add(Flatten())
# add fully-connected layer (i.e., every neuron in the dense layer takes the input from all other neurons of the previous layer)
model.add(Dense(64, activation='relu'))
# add dropout layer (use rate between 0.2 to 0.5) to reduce overfitting by randomly setting inputs to 0
model.add(Dropout(0.5))
# output layer; 'sigmoid' for binary classification ('softmax' for multi-classification)
model.add(Dense(1, activation='sigmoid'))
# show model summary
model.summary()

## Train the model
print('''\nLet's train the model''')
# use Adam (A Method for Stochastic Optimization) optimizer (combines RMSprop and AdaGrad),
# binary cross entropy as the error of the classification because of two classes
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# monitor validation loss and stop training when increment observed; restore weights from the end of the best epoch
earlystopping = EarlyStopping(monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True, verbose=2)
model_train = model.fit(train_set, epochs=20, validation_data=val_set, callbacks =[earlystopping], verbose=2)

## Test the model
print('\nNow test the model')
test_score = model.evaluate(test_set, verbose=1)
print('Test accuracy:', test_score[1])

## Plot few images for visual inspection
# pneumonia_detection.show_imgs(10)