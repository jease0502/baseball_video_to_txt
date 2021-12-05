import numpy as np
from cv2 import cv2
import tensorflow as tf

chars = '0123456789'
char_to_num = {c: i for i, c in enumerate(chars)}
num_to_char = {i: c for i, c in enumerate(chars)}


def resize_pad(img, img_shape):
    h, w, c = img.shape
    height, width, _ = img_shape
    ratio_h, ratio_w = height / h, width / w
    ratio = min(ratio_h, ratio_w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    new_img = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[(height - new_h) // 2:(height - new_h) // 2 + new_h,
           (width - new_w) // 2:(width - new_w) // 2 + new_w, :] = new_img
    return canvas


def sparse_tuple_from(sequences):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(
        indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def dense_to_sparse_info(arr):
    arr = np.array(arr)
    indices_y, indices_x = np.where(arr != -1)
    indices_y = np.expand_dims(indices_y, axis=-1)
    indices_x = np.expand_dims(indices_x, axis=-1)
    indices = np.hstack([indices_y, indices_x])

    values = arr[indices_y, indices_x].reshape(-1)
    shape = arr.shape
    return indices, values, shape


def preprocess_label(sparse_label):
    shape = tf.shape(sparse_label)
    label = tf.sparse.to_dense(sparse_label, default_value=-1)
    label = tf.reshape(label, (shape[0], shape[-1]))

    indices, values, dense_shape = dense_to_sparse_info(label)
    sparse_label = tf.SparseTensor(indices, values, dense_shape)
    sparse_label = tf.cast(sparse_label, tf.int32)

    return label, sparse_label


def parse_output(logits):
    def sparse2dense(tensor, shape):
        tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor = tf.cast(tensor, tf.float32)
        return tensor

    batch_size = tf.shape(logits)[0]
    max_width = tf.shape(logits)[1]
    logit_length = tf.fill([batch_size], max_width)
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length)
    y_pred = sparse2dense(decoded[0], [batch_size, max_width])
    return y_pred


def postprocess_remove_blank(pred, blank=-1):
    pred = np.array(pred, np.int32)
    results = []
    for p in pred:
        p_no_blank = p[np.where(p != -1)[0]]
        result = ''.join([num_to_char[n] for n in p_no_blank])

        results.append(result)
    return results
