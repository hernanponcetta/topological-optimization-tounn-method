import csv
from datetime import datetime

import numpy as np
import torch.nn as nn
from fenics import cells


# Helper function to obtain a unix timestamp
def create_time_stamp():
    return datetime.timestamp(datetime.now())


# Helper function to extract mid points for each cell
def create_mid_points(mesh, dim):
    mid_points = [cell.midpoint().array()[:] for cell in cells(mesh)]
    mid_points = np.array([row[:dim] for row in mid_points])
    return mid_points


# Helper function initializer weights of the neural network
def weight_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)


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
                'loss': row[3],
                'relation_grey_elements': row[4]
            })


# Helper function to write training parameters to file
def write_optimization_data(name, data_directory, data):
    data_directory = "{data_directory}/parameters.txt".format(data_directory=data_directory)

    with open(data_directory, 'w') as f:
        f.write("Topology optmization parameters for {name}\n".format(name=name))
        f.write("\n")

        for key in data:
            f.write("{key}: {data}\n".format(key=key, data=data[key]))