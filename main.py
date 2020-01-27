
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
import numpy as np
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import json
import pandas as pd
import sqlite3

class DB():
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = lambda cursor, row: row if len(row) > 1 else row[0]
        self.c = self.conn.cursor()

    def get_train_ids(self):
        result = self.c.execute('select * from trains').fetchall()
        # result = self.c.execute('select ImageId from test_image').fetchall()
        return result

    def get_val_ids(self):
        result = self.c.execute('select * from vals').fetchall()
        # result = self.c.execute('select ImageId from test_image').fetchall()
        return result

    def get_masks(self, img_id):
        result = self.c.execute("select * from masks where ImageId = '{}'".format(img_id)).fetchall()
        return result



# define a configuration for the model
class ClConfig(Config):
    # Give the configuration a recognizable name
    NAME = "conf"
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 46
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    VALIDATION_STEPS = 50 

    BACKBONE = "resnet50"    

    # USE_MINI_MASK = False

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (256, 256)

    MAX_GT_INSTANCES = 50

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

# class that defines and loads the dataset
class ClDataset(utils.Dataset):

    m_dataset_dir = 'data/'

    database_name = 'data.db'

    # with open(m_dataset_dir+'validation_img_ids.json') as file:
    #     validation_ids = json.load(file)

    # _m_df = pd.read_csv(m_dataset_dir + "train.csv")

    def __init__(self, mode):
        self.mode = mode

        self.db = DB(ClDataset.database_name)

        utils.Dataset.__init__(self)

    def rle_decode(self, rle, shape):
        """Decodes an RLE encoded list of space separated
        numbers and returns a binary mask."""
        rle = list(map(int, rle.split()))
        rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
        rle[:, 1] += rle[:, 0]
        rle -= 1
        mask = np.zeros([shape[0] * shape[1]], np.bool)
        for s, e in rle:
            assert 0 <= s < mask.shape[0]
            assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
            mask[s:e] = 1
        # Reshape and transpose
        mask = mask.reshape([shape[1], shape[0]]).T
        return mask

    # load the dataset definitions
    def load_clothes(self):
        data = json.load(open(ClDataset.m_dataset_dir + "label_descriptions.json"))
        # define one class
        for i in range(ClConfig.NUM_CLASSES-1):
            self.add_class("dataset", data["categories"][i]['id'], data["categories"][i]['name'])
        # define data locations
        images_dir = ClDataset.m_dataset_dir + 'train/'

        if self.mode == 'train':
            img_ids = self.db.get_train_ids()
        elif self.mode == 'val':
            img_ids = self.db.get_val_ids()

        # find all images
        for image_id in img_ids:
            img_path = images_dir + image_id
            self.add_image('dataset', image_id=image_id, path=img_path)

    # load the masks for an image
    def load_mask(self, fucked_up_image_id):
        masks = list()
        classes = list()

        info = self.image_info[fucked_up_image_id]
        fine_image_id = info['id']

        current_image_data = self.db.get_masks(fine_image_id)
        for i in current_image_data:
            enc_pixels = i[2]
            size = (i[3], i[4])
            class_id = i[5].split('_')[0]

            # masks.append(self.rle_decode(enc_pixels, size))
            decoded = self.rle_decode(enc_pixels, size)
            masks.append(decoded)
            classes.append(int(class_id))
        # get details of image
        # masks = np.stack(masks, axis=-1)
        return np.stack(masks, axis=2).astype(np.bool), np.array(classes, dtype=np.int32)

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

if __name__ == '__main__':
    # prepare config
    config = ClConfig()

    # load the train dataset
    train_set = ClDataset('train')
    train_set.load_clothes()
    train_set.prepare()

    val_set = ClDataset('val')
    val_set.load_clothes()
    val_set.prepare()

    model = MaskRCNN(mode='training', model_dir='./', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

    model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=40, layers='all')