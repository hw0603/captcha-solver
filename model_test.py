from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import os

# Batch size for training and validation
batch_size = 25

# Desired image dimensions
img_width = 150
img_height = 40

prediction_model = load_model('data.h5')
# prediction_model.summary()
max_length = 4

# Path to the data directory
data_dir = Path("./testimage/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
# characters = set(char for label in labels for char in label)
# characters = ['k', 'y', 'f', '3', '4', '5', '6', 'r', 'h', 'n', 'b', 'p', 'e', 'd', 'c', 'g', 'w', 'm', 'x', '8', '2', '7', 'a']
characters = "kyf3456rhnbpedcgwmx827a_.ijloqstuvz019"
# characters = "abcdefghijklmnopqrstuvwxyz0123456789_. "

# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.1, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    # train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    # x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[:]], labels[indices[:]]
    return x_valid, y_valid


def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


# Splitting data into training and validation sets
x_valid, y_valid = split_data(np.array(images), np.array(labels))

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text



#  Let's check results on some validation samples
for batch in validation_dataset.take(1):
    batch_images = batch["image"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    _, ax = plt.subplots(5, 5, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 5, i % 5].imshow(img, cmap="gray")
        ax[i // 5, i % 5].set_title(title)
        ax[i // 5, i % 5].axis("off")
plt.show()

# 파일명 변경
# for batch in validation_dataset:
#     batch_images = batch["image"]
#     batch_labels = batch["label"]

#     preds = prediction_model.predict(batch_images)
#     pred_texts = decode_batch_predictions(preds)

#     orig_texts = []
#     for label in batch_labels:
#         label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
#         orig_texts.append(label)

#     for filename, captcha in zip(orig_texts, pred_texts):
#         src = os.path.join("./testimage", f"{filename}.png")
#         dst = os.path.join("./testimage", f"{captcha}.png")
#         try:
#             os.rename(src, dst)
#         except FileExistsError:
#             print("파일이 이미 존재합니다.")
#             os.remove(src)
#         print(f"{src} -> {dst}")
