# Note the code in this file is based heavily off of the re-used code in the CONTENT Repository Found
# here: https://github.com/danicaxiao/CONTENT
# Although some variables and the way it is run is changed for better interpretability/ease of use

import sys
import os
import psutil
import time
import DataPrep as dp
from Configuration import Configuration
from PatientDataLoader import PatientDataLoader
import CONTENT as content
import CONTENT_FixedBatch as contentFixed


if __name__ == '__main__':
    fixed = (len(sys.argv) > 1) and "fixed" in sys.argv
    new = (len(sys.argv) > 1) and "new" in sys.argv

    # First Let's Get Our Data Prepped
    # If this is a new input file with new splits, we need to run this, otherwise we can skip
    if new or not os.path.exists("data/X_train.pkl"):
        data = dp.retrieve_data(False) # Retrieves Input Data If it doesn't already exist
        dp.data_to_csv(data) # Writes vocab and stop files based on Input
        word2ind = dp.load_data_from_file() # Loads vocab from csv files into word2ind vector
        events = dp.extract_events() # Extracts Events From Input File
        data, labels = dp.convert_format(word2ind, events) # Converts Data into More Useful Format
        dp.splits(data, labels) # Creates Training, Validation, and Testing Splits and Writes them to Pickled Files

    # Now We Need to Setup our DataLoader and Config for the Model
    FLAGS = Configuration()
    data_set = PatientDataLoader(FLAGS)
    iterator = data_set.iterator()

    if fixed:
        thetaPath = "theta/thetas1.npy"
        start = time.time()
        contentFixed.run(data_set)
        end = time.time()
        contentFixed.clustering(thetaPath, data_set)
        contentFixed.eval(1)
    else:
        thetaPath = "theta_with_rnnvec/thetas_train0.npy"
        start = time.time()
        content.run(data_set)
        end = time.time()
        # content.eval(2)
        # content.clustering(thetaPath, data_set)

    print("Total GPU Time to Run Experiment: {}".format(end - start))
    process = psutil.Process(os.getpid())
    print("Total Memory Usage in Process: {} MB".format(process.memory_info().rss / 1024)) 