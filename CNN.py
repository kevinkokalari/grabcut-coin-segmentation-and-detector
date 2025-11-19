
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

IMG_SZ   = 224
BATCH    = 10
EPOCHS   = 15
DATA_DIR = Path("data")

#SOLVES A PROBLEM RELATED TO A LIBRARY AND ARM VERSIONS OF MACOS.
opts = tf.config.optimizer.get_experimental_options()
opts["disable_meta_optimizer"] = True
tf.config.optimizer.set_experimental_options(opts)
# --------------------------------------------------------

train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1,
    brightness_range=[0.5,1.5],
    horizontal_flip=True, fill_mode='nearest')
val_gen   = ImageDataGenerator(rescale=1/255.)


train = train_gen.flow_from_directory(
    DATA_DIR/"train", target_size=(IMG_SZ,IMG_SZ),
    batch_size=BATCH, class_mode='categorical')
val   = val_gen.flow_from_directory(
    DATA_DIR/"val", target_size=(IMG_SZ,IMG_SZ),
    batch_size=BATCH, class_mode='categorical',
    shuffle=True)

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SZ,IMG_SZ,3), include_top=False,
    weights='imagenet', pooling='avg')
base.trainable = False         # freeze Imagenet layers

x = layers.Dense(384, activation='relu')(base.output)
out = layers.Dense(train.num_classes, activation='softmax')(x)
model = models.Model(base.input, out)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train, epochs=EPOCHS, validation_data=val)

# fine-tune last 50 layers
for layer in base.layers[-50:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=8, validation_data=val)

model.save("model/yen_cnn.h5")
print("Saved to model/yen_cnn.h5")