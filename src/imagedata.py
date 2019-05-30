import csv
import numpy as np
from random import shuffle

class ImageData:

    def __init__ \
    (
        self,
        img_dir,
        attr_file,
        image_size,
        batch_size,
        selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
        validation_ratio = 0.1
    ):
        self.img_dir = img_dir
        self.attr_file = attr_file
        self.image_size = image_size
        self.batch_size = batch_size
        self.selected_attrs = selected_attrs
        self.validation_ratio = validation_ratio

        # stores conversion between attributes idx <-> string.
        self.attr_indices = {}
        self.all_attrs = []

        # data sets [img_name, [labels (0/1)]]
        self.train_set = []
        self.validation_set = []
        self.prepare_labels()

    # returns number of batches in the set
    def __len__(self):
        return len(self.train_set) // self.batch_size

    # returns a batch
    def __getitem__(self, idx):
        if type(idx) != int:
            raise("index operator expects integer")

        return self.train_set \
        [
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

    def image_dim(self):
        return (self.image_size[0], self.image_size[1], 3 + len(self.selected_attrs))

    def disc_dim(self):
        return (self.image_size[0], self.image_size[1], 3)

    def num_classes(self):
        return len(self.selected_attrs)

    # Reads labels from the attribute file.
    # Then generates the train / validation split 
    def prepare_labels(self):
        rows = []
        with open(self.attr_file, "r") as file:
            reader = csv.reader(file)

            # reads header line first, generate indices etc..
            self.prepare_attr_info(reader.__next__())

            for row in reader:
                # convert strings to numbers
                labels = []
                for idx in range(1, len(row)):
                    labels.append(1 if row[idx] == "1" else -1)
                rows.append([row[0], labels])

        # validation is 0.1 total data
        shuffle(rows)
        pivot = int(self.validation_ratio * len(rows))
        self.validation_set = rows[:pivot]
        self.train_set = rows[pivot:]

        if len(self.selected_attrs) == 0:
            self.selected_attrs = self.all_attrs

    # generates attribute conversion structures.
    # Also validates selected labels
    def prepare_attr_info(self, attributes):
        attributes = attributes[1:]

        for idx, attr in enumerate(attributes):
            self.attr_indices[attr] = idx
            self.all_attrs.append(attr)
        
        for attr in self.selected_attrs:
            if not attr in self.attr_indices:
                raise Exception("Unknown label: {}".format(attr))