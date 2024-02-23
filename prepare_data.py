# coding: utf-8
"""
    Building the dataset trying to follow `glyphreader` repository
    as closely as possible; dataset is constructed and split using Pool there,
    so the exactly the same split as in `glyphreader` does not seem
    to be reproducible; hence we do our own splitting.
"""

import urllib.request
import hashlib
import shutil
import logging
from collections import Counter
import os
from os.path import join, isdir, isfile
from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import train_test_split

UNKNOWN_LABEL = "UNKNOWN"


def main():
    ap = ArgumentParser()
    ap.add_argument(
        "--data_path", default="/".join(("data", "Dataset", "Manual", "Preprocessed"))
    )
    ap.add_argument("--prepared_data_path", default="prepared_data")
    ap.add_argument("--test_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=261)

    arguments = ap.parse_args()

    # prepare yourself for some hardcode
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    download_dataset()

    file_dir = os.path.dirname(__file__)
    stele_path = os.path.join(file_dir, arguments.data_path)
    steles = filter(isdir, map(lambda f: join(stele_path, f), os.listdir(stele_path)))

    res_image_paths, labels = [], []
    for stele in steles:
        image_paths = filter(isfile, map(lambda f: join(stele, f), os.listdir(stele)))

        for path in image_paths:
            res_image_paths.append(path)
            labels.append(path[(path.rfind("_") + 1) : path.rfind(".")])

    list_of_paths = np.asarray(res_image_paths)
    labels = np.array(labels)

    logging.debug(f"Labels total: {len(set(labels))}")

    labels_just_once = np.array([l for (l, c) in Counter(labels).items() if c <= 1])
    logging.debug(f"Labels seen just once: {len(labels_just_once)}")

    # those hieroglyphs that were seen in data only once, go to TRAIN set
    to_be_added_to_train_only = np.nonzero(np.isin(labels, labels_just_once))[0]

    # the hieroglyphs that have NO label are to be removed
    to_be_deleted = np.nonzero(labels == UNKNOWN_LABEL)[0]

    # we remove all elements of these two sets
    to_be_deleted = np.concatenate([to_be_deleted, to_be_added_to_train_only])
    filtered_list_of_paths = np.delete(list_of_paths, to_be_deleted, 0)
    filtered_labels = np.delete(labels, to_be_deleted, 0)

    # we split the data
    train_paths, test_paths, y_train, y_test = train_test_split(
        filtered_list_of_paths,
        filtered_labels,
        stratify=filtered_labels,
        test_size=arguments.test_fraction,
        random_state=arguments.seed,
    )

    # we add the 'single-occurence' folks to the train set
    train_paths = np.concatenate(
        [train_paths, list_of_paths[to_be_added_to_train_only]]
    )
    y_train = np.concatenate([y_train, labels[to_be_added_to_train_only]])

    # Delete directory if already exists
    if os.path.exists(arguments.prepared_data_path):
        shutil.rmtree(arguments.prepared_data_path)

    # then we copy all
    os.makedirs(arguments.prepared_data_path)
    for label in set(y_train):
        os.makedirs(join(arguments.prepared_data_path, "train", label))
    for label in set(y_test):
        os.makedirs(join(arguments.prepared_data_path, "test", label))

    copy_to_prepared_data_dir(train_paths, y_train, arguments.prepared_data_path)
    copy_to_prepared_data_dir(test_paths, y_test, arguments.prepared_data_path)


def generate_file_id(filepath):
    return hashlib.md5(filepath.encode("utf-8")).hexdigest() + ".png"


def copy_to_prepared_data_dir(paths, dataset, prepared_data_path):
    for filepath, label in zip(paths, dataset):
        filename = join(
            prepared_data_path, "test", label, filepath, generate_file_id(filepath)
        )
        shutil.copyfile(filepath, filename)


def download_dataset():
    url = "http://iamai.nl/downloads/GlyphDataset.zip"
    zip_file_path = "GlyphDataset.zip"
    extracted_dir = "data"

    urllib.request.urlretrieve(url, zip_file_path)
    logging.debug("Downloaded GlyphDataset.zip")

    shutil.unpack_archive(zip_file_path, extracted_dir)
    logging.debug("Unzipped GlyphDataset.zip")

    os.remove(zip_file_path)


if __name__ == "__main__":
    main()
