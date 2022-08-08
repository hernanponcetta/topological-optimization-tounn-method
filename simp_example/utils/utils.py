import csv
from datetime import datetime

import numpy as np
import torch.nn as nn
from fenics import cells


# Helper function to obtain a unix timestamp
def create_time_stamp():
    return datetime.timestamp(datetime.now())


# Helper function to write training data to csv file
def write_to_csv(training_data, data_directory):
    data_directory = "{data_directory}/training_data.csv".format(data_directory=data_directory)

    with open(data_directory, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'objective', 'density_avg', 'loss', 'relation_grey_elements']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in training_data:
            writer.writerow({
                'epoch': row[0],
                'objective': row[1],
                'density_avg': row[2],
            })