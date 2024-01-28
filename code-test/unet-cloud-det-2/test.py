import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf

os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

height = 384
width = 384

dataset_path = "test"
save_path = "prediction/non-aug"
model_path = "files/non-aug/unet-checkpoint-3.h5"

model = tf.keras.models.load_model(model_path)
model.summary()

test_x = sorted(glob(os.path.join(dataset_path, "images", "*")))
print(f"Test Images: {len(test_x)}")

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
    print(name)
    cv2.imwrite(os.path.join(save_path, name), p)
    cv2.imshow("isa", p)
    cv2.waitKey(5)
    
mean_time = np.mean(time_taken)
mean_fps = 1/mean_time
print(f"Mean Time: {mean_time: 1.5f} - Mean FPS: {mean_fps:2.5f}")
