import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl

ATT_ID = {'has_primary_color_blue': 0, 'has_primary_color_brown': 1, 'has_primary_color_black': 2, 'has_primary_color_grey': 3, 'has_primary_color_white': 4,'has_underparts_color_white': 5, 'has_underparts_color_grey': 6, 'has_underparts_color_blue': 7, 'has_underparts_color_black': 8, 'has_underparts_color_brown': 9,'has_upperparts_color_white': 10, 'has_upperparts_color_black': 11, 'has_upperparts_color_grey': 12, 'has_upperparts_color_blue': 13, 'has_upperparts_color_brown': 14}
ID_ATT = {v: k for k, v in ATT_ID.items()}

def make_celeba_dataset(img_dir,
                        label_path,
                        att_names,
                        batch_size,
                        load_size=286,
                        crop_size=256,
                        training=True,
                        drop_remainder=True,
                        shuffle=True,
                        repeat=1):
    img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
    print('label_path: ')
    print(label_path)
    print('len(ATT_ID.keys()): ')
    print(len(ATT_ID.keys()))
    img_paths = np.array([py.join(img_dir, img_name) for img_name in img_names])
    print('img_paths: ')
    print(img_paths)
    labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, len(ATT_ID.keys()) + 1))
    labels = labels[:, np.array([ATT_ID[att_name] for att_name in att_names])]
    print('len(labels) :')
    print(len(labels))
    print('labels[1]: ')
    print(labels[1])

    if shuffle:
        idx = np.random.permutation(len(img_paths))
        img_paths = img_paths[idx]
        labels = labels[idx]

    if training:
        def map_fn_(img, label):
            img = tf.image.resize(img, [crop_size, crop_size])
            # img = tl.random_rotate(img, 5)
            img = tf.image.random_flip_left_right(img)
            # img = tf.image.random_crop(img, [crop_size, crop_size, 3])
            # img = tl.color_jitter(img, 25, 0.2, 0.2, 0.1)
            # img = tl.random_grayscale(img, p=0.3)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label
    else:
        def map_fn_(img, label):
            img = tf.image.resize(img, [crop_size, crop_size])
            # img = tl.center_crop(img, size=crop_size)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          labels=labels,
                                          drop_remainder=drop_remainder,
                                          map_fn=map_fn_,
                                          shuffle=shuffle,
                                          repeat=repeat)

    if drop_remainder:
        len_dataset = len(img_paths) // batch_size
    else:
        len_dataset = int(np.ceil(len(img_paths) / batch_size))

    return dataset, len_dataset


def check_attribute_conflict(att_batch, att_name, att_names):
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    idx = att_names.index(att_name)
    
    for att in att_batch:
        if att_name in ['has_primary_color_blue', 'has_primary_color_brown', 'has_primary_color_black', 'has_primary_color_grey', 'has_primary_color_white'] and att[idx] == 1:
            for n in ['has_primary_color_blue', 'has_primary_color_brown', 'has_primary_color_black', 'has_primary_color_grey', 'has_primary_color_white']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['has_underparts_color_white', 'has_underparts_color_grey', 'has_underparts_color_blue', 'has_underparts_color_black', 'has_underparts_color_brown'] and att[idx] == 1:
            for n in ['has_underparts_color_white', 'has_underparts_color_grey', 'has_underparts_color_blue', 'has_underparts_color_black', 'has_underparts_color_brown']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['has_upperparts_color_white', 'has_upperparts_color_black', 'has_upperparts_color_grey', 'has_upperparts_color_blue', 'has_upperparts_color_brown'] and att[idx] == 1:
            for n in ['has_upperparts_color_white', 'has_upperparts_color_black', 'has_upperparts_color_grey', 'has_upperparts_color_blue', 'has_upperparts_color_brown']:
                if n != att_name:
                    _set(att, 0, n)
       
    return att_batch
