import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

lr = 1e-4
height = 384
width = 384
files_dir = os.path.join("files")
model_file = os.path.join(files_dir, "unet-checkpoint.h5")
log_file = os.path.join(files_dir, "log.csv")

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2,2))(x)
    return x, p

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 512)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model

def load_data(train_count, val_count):
    train_x = sorted(glob(os.path.join("train_blue", "*")))[:train_count]
    train_y = sorted(glob(os.path.join("train_gt", "*")))[:train_count]

    valid_x = sorted(glob(os.path.join("train_blue", "*")))[train_count:train_count+val_count]
    valid_y = sorted(glob(os.path.join("train_gt", "*")))[train_count:train_count+val_count]

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([height, width, 3])
    y.set_shape([height, width, 1])

    return x, y

def tf_dataset(x, y, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

input_shape = (height, width, 3)

callbacks= [
    ModelCheckpoint(model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4),
    CSVLogger(log_file),
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False)
]

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

binary_iou = tf.keras.metrics.BinaryIoU(name='IoU')
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
epochs = 30
batch_size = 2
train_count = 1000
val_count = 300

(train_x, train_y), (valid_x, valid_y) = load_data(train_count, val_count)
train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

model = build_unet(input_shape)
opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc", binary_iou, recall, precision])

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=callbacks
)



import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from google.colab.patches import cv2_imshow

os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

# %matplotlib inline

height = 384
width = 384

model_path = "files/unet-checkpoint-2.h5"
model = tf.keras.models.load_model(model_path)

save_path = "predict/"

test_x = sorted(glob(os.path.join("test", "*")))
print(test_x)

time_taken = []
i = 0
for x in tqdm(test_x):
    name = f"{i}.png"
    i += 1

    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = x/255.0
    x = np.expand_dims(x, axis=0)

    start_time = time.time()
    p = model.predict(x)[0]
    total_time = time.time()-start_time
    time_taken.append(total_time)

    p = p>0.5
    p = p*255.0

    # cv2.imwrite(f"{save_path}/{name}", p)
    print(test_x[i-1])
    cv2.imwrite(os.path.join(save_path, name), p)
    cv2_imshow(p)

mean_time = np.mean(time_taken)
mean_fps = 1/mean_time
print(f"Mean Time: {mean_time: 1.5f} - Mean FPS: {mean_fps:2.5f}")