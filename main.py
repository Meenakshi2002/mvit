import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mobilevit import *

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to the [0, 1] range
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    horizontal_flip=True,  # Randomly flip images horizontally
    zoom_range=0.2  # Randomly zoom in on images
)

# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(256, 256),  
    batch_size=32,
    class_mode='categorical'
)

valid_datagen = ImageDataGenerator(     
    rescale=1.0/255)

valid_generator = valid_datagen.flow_from_directory(
    'data/test',
    target_size=(256, 256),#test  
    batch_size=32,
    class_mode='categorical'
)

model = MobileViT_S(input_shape=(256, 256, 3), num_classes=4)  # Updated input_shape to (224, 224, 3)
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=valid_generator
)

model.save('model2.h5')
print('model saved')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(valid_generator)
