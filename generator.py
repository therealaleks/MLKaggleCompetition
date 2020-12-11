import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import h5py

from multiprocessing import Pool, Manager, cpu_count
from functools import partial

COUNT = 10
DEBUG = True
MULTI = False
mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']


def check_mnist_dir(data_dir):
    downloaded = np.all([os.path.isfile(os.path.join(data_dir, key)) for key in mnist_keys])
    if not downloaded:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_mnist(data_dir)
    else:
        print('MNIST was found')


def download_mnist(data_dir):
    data_url = 'http://yann.lecun.com/exdb/mnist/'
    for k in mnist_keys:
        k += '.gz'
        url = (data_url + k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gunzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)


def extract_mnist(data_dir):
    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1))

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((num_mnist_train)))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1))

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((num_mnist_test)))

    return np.concatenate((train_image, test_image)), \
           np.concatenate((train_label, test_label))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def generator(images, labels):
    indexes = np.arange(70000)
    images = np.array([cv2.resize(img, (12, 12)) for img in images])
    generated_images = []
    generated_labels = []

    for _ in range(COUNT):

        digit_count = random.randint(1, 5)

        if digit_count == 1:
            offset = random.randint(23, 28)
        elif digit_count == 2:
            offset = random.randint(18, 23)
        elif digit_count == 3:
            offset = random.randint(13, 18)
        elif digit_count == 4:
            offset = np.random.randint(8, 13)
        elif digit_count == 5:
            offset = random.randint(5, 8)
        else:
            raise ValueError("digits should be between 1 and 5 inclusive")

        np.random.shuffle(indexes)
        idx_choices = indexes[:digit_count]
        instance_label = np.concatenate((labels[idx_choices], np.array([10] * (5 - digit_count)))).astype(np.uint8)
        instance_image = np.zeros((64, 64)).astype(np.uint8)

        for img in images[idx_choices]:
            img = rotate_image(img, random.randint(-30, 30))
            img = np.maximum(instance_image[26:38, offset: offset + 12].reshape(-1),
                             img.reshape(-1))
            instance_image[26:38, offset: offset + 12] = img.reshape(12, 12)
            offset += random.randint(8, 11)

        if DEBUG:
            print(instance_label)
            plt.matshow(instance_image)
            plt.show()

        generated_images.append(instance_image)
        generated_labels.append(instance_label)

    return np.array(generated_images), np.array(generated_labels)


def helper(count, images, labels, res):

    indexes = np.arange(70000)

    for _ in range(count):

        if _ % 10000 == 0:
            print(_)

        digit_count = random.randint(1, 5)

        if digit_count == 1:
            offset = random.randint(23, 28)
        elif digit_count == 2:
            offset = random.randint(18, 23)
        elif digit_count == 3:
            offset = random.randint(13, 18)
        elif digit_count == 4:
            offset = np.random.randint(8, 13)
        elif digit_count == 5:
            offset = random.randint(5, 8)
        else:
            raise ValueError("digits should be between 1 and 5 inclusive")

        idx_choices = indexes[:digit_count]
        instance_label = np.concatenate((labels[idx_choices], np.array([10] * (5 - digit_count)))).astype(np.uint8)
        instance_image = np.zeros((64, 64)).astype(np.uint8)

        for img in images[idx_choices]:
            img = rotate_image(img, random.randint(-30, 30))
            img = np.maximum(instance_image[26:38, offset: offset + 12].reshape(-1),
                             img.reshape(-1))
            instance_image[26:38, offset: offset + 12] = img.reshape(12, 12)
            offset += random.randint(8, 11)

        res.append((instance_image, instance_label))


def multiproc_generator(images, labels, cores):
    results = None
    generated_images = None
    generated_labels = None
    images = np.array([cv2.resize(img, (12, 12)) for img in images])

    with Manager() as manager, Pool(processes=cores) as pool:
        results = manager.list()
        f = partial(helper, images=images, labels=labels, res=results)
        for _ in pool.imap_unordered(f, [COUNT//cores] * cores):
            pass
        results = list(results)
        generated_images = np.array([tup[0] for tup in results])
        generated_labels = np.array([tup[1] for tup in results])

    return generated_images, generated_labels


if __name__ == "__main__":

    check_mnist_dir("datasets")
    images, labels = extract_mnist("datasets")
    if MULTI:
        cores = cpu_count() - 1 or 1
        generated_images, generated_labels = multiproc_generator(images, labels, cores)
        COUNT = (COUNT // cores) * cores
    else:
        generated_images, generated_labels = generator(images, labels)

    with h5py.File("generated.hdf5", "w") as f:
        fdset = f.create_dataset("generated_train_features", (COUNT, 64, 64), dtype="i")
        fdset[...] = generated_images
        ldset = f.create_dataset("generated_train_labels", (COUNT, 5), dtype="i")
        ldset[...] = generated_labels

    print(f"Generated {COUNT} images.")

