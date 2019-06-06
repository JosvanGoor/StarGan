import csv
import numpy as np
from random import shuffle
from tqdm import tqdm
from math import floor

POSSIBLE_HAIR_COLORS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]

class ImageData:

    def __init__ \
    (
        self,
        image_dir,
        attribute_file,
        image_size,
        batch_size,
        selected_attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
        validation_ratio = 0.1
    ):
        self.image_dir = image_dir
        self.attribute_file = attribute_file
        self.image_size = image_size
        self.batch_size = batch_size
        self.selected_attributes = selected_attributes
        self.validation_ratio = validation_ratio

        self.attribute_indices = {}
        self.all_attributes = []
        
        # [filename, selected_labels, all_labels]
        self.validation_set = []
        self.train_set = []

    '''
        Methods for accessing image data
    '''

    def trainbatch_count(self):
        return floor(len(self.train_set) / self.batch_size)

    def get_train_batch(self, index):
        return self.train_set \
            [
                index * self.batch_size
                : (index + 1) * self.batch_size
            ]

    '''
        Initializes the inner state.
    '''
    def read_attribute_file(self):
        rows = []
        with open(self.attribute_file, "r") as file:
            reader = csv.reader(file)

            self.prepare_labels(reader.__next__())

            for row in tqdm(reader, desc = "Reading label data"):
                labels = []
                for idx in range(1, len(row)):
                    labels.append(1 if row[idx] == "1" else 0)
                rows.append([row[0], self.selected_labels(labels), labels])

        # split between test and validation
        # shuffle(rows)
        pivot = int(self.validation_ratio * len(rows))
        self.validation_set = rows[:pivot]
        self.train_set = rows[pivot:]

        if (len(self.selected_attributes) == 0):
            self.selected_attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]

    def selected_labels(self, labelset):
        labels = []

        # we assume here that the original labels only have 1 hair color selected always
        for label in self.selected_attributes:
            label_index = self.attribute_indices[label]
            labels.append(labelset[label_index])
        
        return labels

    def prepare_labels(self, attributes):
        self.all_attributes = attributes[1:]

        for idx, attr in enumerate(self.all_attributes):
            self.attribute_indices[attr] = idx
            
        for attr in self.selected_attributes:
            if not attr in self.attribute_indices:
                raise Exception("Unknown label: '{}'".format(attr))