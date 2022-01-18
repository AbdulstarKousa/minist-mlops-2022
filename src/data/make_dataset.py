# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # ====== Test Data
    # load
    test_imgs = np.load(os.path.join(input_filepath, "test.npz"))["images"]
    test_labs = np.load(os.path.join(input_filepath, "test.npz"))["labels"]

    # to tensor
    test_imgs = torch.Tensor(test_imgs)
    test_labs = torch.Tensor(test_labs).type(torch.LongTensor)
    # normalize
    test_imgs = torch.nn.functional.normalize(test_imgs)
    # save
    logger.info("saving the testing data")
    print(test_imgs.shape)
    torch.save(test_imgs, os.path.join(output_filepath, "test_images.pt"))
    torch.save(test_labs, os.path.join(output_filepath, "test_labels.pt"))

    # ====== Train Data
    # load
    filenames = [
        "train_0.npz",
        "train_1.npz",
        "train_2.npz",
        "train_3.npz",
        "train_4.npz",
    ]
    train_imgs_lst = []
    train_labs_lst = []
    for filename in filenames:
        train_imgs_lst.append(np.load(os.path.join(input_filepath, filename))["images"])
        train_labs_lst.append(np.load(os.path.join(input_filepath, filename))["labels"])
    train_imgs = np.concatenate(tuple(train_imgs_lst))
    train_labs = np.concatenate(tuple(train_labs_lst))
    # to tensor
    train_imgs = torch.Tensor(train_imgs)
    train_labs = torch.Tensor(train_labs).type(torch.LongTensor)
    # normalize
    train_imgs = torch.nn.functional.normalize(train_imgs)
    # save
    logger.info("saving the training data")
    print(train_imgs.shape)
    torch.save(train_imgs, os.path.join(output_filepath, "train_images.pt"))
    torch.save(train_labs, os.path.join(output_filepath, "train_labels.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


# to run
# python src/data/make_dataset.py data/raw data/processed
