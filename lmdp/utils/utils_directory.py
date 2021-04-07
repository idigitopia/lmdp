#todo clean this file
import logging
import os
import pickle as pk
import shutil

import numpy as np

logger = logging.getLogger("lmdp_logger")


def removeDirectory(path, verbose=True):
    if (os.path.isdir(path)):
        if (True):  # input("are you sure you want to remove this directory? (Y / N): " + path) == "Y" ):
            shutil.rmtree(path)
    else:
        if (verbose):
            logger.debug("No Directory to be romved")


def makeDirectory(path, verbose=True):
    try:
        os.mkdir(path)
    except OSError:
        if (verbose):
            logger.debug("Creation of the directory %s failed" % path)
    else:
        if (verbose):
            logger.debug("Successfully created the directory %s " % path)


def resetParentDirectory(path, verbose=False):
    path = '/'.join(path.rstrip("/").split("/")[:-1])
    removeDirectory(path, verbose)
    makeDirectory(path, verbose)


def resetDirectory(path, verbose=False):
    removeDirectory(path, verbose)
    makeDirectory(path, verbose)


def resetandseed(parent_dir = None, dirList =None ):
    resetDirectory(parent_dir)

    for dir in dirList:
        makeDirectory(dir)

def cache_data(data, file_path):
    with open(file_path, 'wb') as file:
        pk.dump(data,file)
    logger.debug("buffer saved to cache:{}".format(file_path) )  # todo add it to logger


def fetch_from_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pk.load(file)
        logger.debug("Successfully fetched from cache:{}".format(file_path)  )# todo add it to logger
        return data
    else:
        logger.debug("Could not Fetch, File Does not Exist: {}".format(file_path))
        return False


def create_hierarchy(folders):
    def create_hierarchy_fn(child_folders, parent_folder):
        logger.debug("creating directory:{}".format(parent_folder))
        makeDirectory(parent_folder)
        if len(child_folders) > 0:
            parent_folder = parent_folder + "/" + child_folders[0]
            return create_hierarchy_fn(child_folders[1:], parent_folder)
        else:
            return
    folders = folders.split("/")
    create_hierarchy_fn(folders[1:], folders[0])

def round_state(arr, precision=2):
    return np.array([round(i, precision) for i in arr])

