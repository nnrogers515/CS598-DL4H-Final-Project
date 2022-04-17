### CODE IN THIS FILE IS BASED ON THE FORMAT OF https://github.com/danicaxiao/CONTENT/blob/master/transform.py
### HOWEVER THE CODE IS WRITTEN IN MY OWN FORM FOR BETTER UNDERSTANDING

import zipfile
import os
import urllib.request
import pandas as pd
import csv
import pickle
import numpy as np

# Global Definitions

# Util Functions From https://github.com/danicaxiao/CONTENT/blob/master/util.py
def save_pkl(path, dump):
    with open(path, 'wb') as file:
        pickle.dump(dump, file)

def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_npy(path, obj):
    np.save(path, obj)

def load_npy(path):
    obj = np.load(path)
    return obj

# Embedding Defs
RARE_WORD = 100
STOP_WORD = 1e4
UNKNOWN = 1

# General File Paths
VOCAB_FILE = "./data/vocab.txt"
STOP_FILE = "./data/stop.txt"
VOCAB_PKL = "./data/vocab.pkl"
SIM_DATA_URL = "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0195024.s001"
SIM_DATA_ZIP = "./txtData.zip"
INPUT_FILE = "./data/S1_File.txt"

# Train/Validation/Test Data File Paths
X_TRAIN_FILE = "./data/X_train.pkl"
X_VALID_FILE = "./data/X_valid.pkl"
X_TEST_FILE = "./data/X_test.pkl"

# Train/Validation/Test Label File Paths
Y_TRAIN_FILE = "./data/Y_train.pkl"
Y_VALID_FILE = "./data/Y_valid.pkl"
Y_TEST_FILE = "./data/Y_test.pkl"

# Train/Validation Split Values
TRAIN_COUNT = 2000
VALID_COUNT = 600
TEST_COUNT = 400

def retrieve_data(print_out=False):
    # Retrieve Data If Not In Our Active Directory
    if not os.path.exists(SIM_DATA_ZIP):
        urllib.request.urlretrieve(SIM_DATA_URL, SIM_DATA_ZIP)

    # Unzip our Data into Usable Form
    if not os.path.exists("./data") or not os.path.exists(INPUT_FILE):
        with zipfile.ZipFile(SIM_DATA_ZIP, 'r') as zipped_file:
            zipped_file.extractall("./data")


    # Read Our Data into a Pandas Table
    data = pd.read_csv(INPUT_FILE, sep='\t', header=0)

    # Check Our Data
    if print_out:
        print("\n", data.head(), "\n")

    return data

def data_to_csv(data, print_out=False):
    # Group Our Data By Description
    desc = data.groupby('DX_GROUP_DESCRIPTION').size().to_frame('SIZE').reset_index()
    rare = desc[desc['SIZE'] > RARE_WORD]
    stop = desc[desc['SIZE'] > STOP_WORD]

    rare = rare.sort_values(by = 'SIZE').reset_index()['DX_GROUP_DESCRIPTION']
    stop = stop.reset_index()['DX_GROUP_DESCRIPTION']
        
    rare.index += 2 # We will follow the studies format of keeping "Unknown" as 1
    
    if print_out:
        print("Writing Vocab List to CSV...")
    rare.to_csv(VOCAB_FILE, sep = '\t', header = False, index = True)
    if print_out:
        print("Done!")
    
    if print_out:
        print("\nWriting Stop Word List to CSV...")
    stop.to_csv(STOP_FILE, sep = '\t', header = False, index = False)
    if print_out:
        print("Done!")
        print("\nData Successfully Written as {} and {} in CSV Format!".format(VOCAB_FILE, STOP_FILE))
    

def load_data_from_file():
    word2ind = {}
    
    with open(VOCAB_FILE, 'r') as vocab_file:
        read_in = csv.reader(vocab_file, delimiter='\t')
        word2ind = { entry[1]:int(entry[0]) for entry in read_in }
        
    # Save Ind2Word Vec to Pickled File
    save_pkl(VOCAB_PKL, {val:key for key, val in word2ind.items()})
    
    return word2ind

# THIS FUNCTION IS DIRECTLY RE-USED FROM https://github.com/danicaxiao/CONTENT/blob/master/transform.py
def convert_format(word_to_index, events, print_out=False):
    # order by PID, DAY_ID
    with open(INPUT_FILE, mode='r') as f:
        # header
        header = f.readline().strip().split('\t')
        if print_out:
            print(header)
        pos = {}
        for key, value in enumerate(header):
            pos[value] = key
        if print_out:
            print(pos)

        docs = []
        doc = []
        sent = []
        labels = []
        label = []

        # init
        line = f.readline()
        tokens = line.strip().split('\t')
        pid = tokens[pos['PID']]
        day_id = tokens[pos['DAY_ID']]
        label.append(tag(events, pid, day_id))

        while line != '':
            tokens = line.strip().split('\t')
            c_pid = tokens[pos['PID']]
            c_day_id = tokens[pos['DAY_ID']]

            # closure
            if c_pid != pid:
                doc.append(sent)
                docs.append(doc)
                sent = []
                doc = []
                pid = c_pid
                day_id = c_day_id
                labels.append(label)
                label = [tag(events, pid, day_id)]
            else:
                if c_day_id != day_id:
                    doc.append(sent)
                    sent = []
                    day_id = c_day_id
                    label.append(tag(events, pid, day_id))

            word = tokens[pos['DX_GROUP_DESCRIPTION']]
            try:
                sent.append(word_to_index[word])
            except KeyError:
                sent.append(UNKNOWN)

            line = f.readline()

        # closure
        doc.append(sent)
        docs.append(doc)
        labels.append(label)

    return docs, labels

# THIS FUNCTION IS DIRECTLY RE-USED FROM https://github.com/danicaxiao/CONTENT/blob/master/transform.py
def tag(events, pid, day_id):
    return 1 if tag_logic(events, pid, day_id) else 0

# THIS FUNCTION IS DIRECTLY RE-USED FROM https://github.com/danicaxiao/CONTENT/blob/master/transform.py
def tag_logic(events, pid, day_id):
    try:
        patient = events.loc[int(pid)]

        # test whether have events within 30 days
        if isinstance(patient, pd.Series):
            return (int(day_id) <= patient.DAY_ID) & (patient.DAY_ID < int(day_id) + 30)

        return patient.loc[(int(day_id) <= patient.DAY_ID) & (patient.DAY_ID < int(day_id) + 30)].shape[0] > 0
    except KeyError:
        # the label is not in the [index]
        return False

# THIS FUNCTION IS DIRECTLY RE-USED FROM https://github.com/danicaxiao/CONTENT/blob/master/transform.py
def extract_events():
    # extract event "INPATIENT HOSPITAL"
    target_event = 'INPATIENT HOSPITAL'

    df = pd.read_csv(INPUT_FILE, sep='\t', header=0)
    events = df[df['SERVICE_LOCATION'] == target_event]

    events = events.groupby(['PID', 'DAY_ID', 'SERVICE_LOCATION']).size().to_frame('COUNT').reset_index()        .sort_values(by=['PID', 'DAY_ID'], ascending=True)        .set_index('PID')

    return events

def splits(X, labels):
    save_pkl(X_TRAIN_FILE, X[:TRAIN_COUNT])
    save_pkl(X_VALID_FILE, X[TRAIN_COUNT:(TRAIN_COUNT + VALID_COUNT)])
    save_pkl(X_TEST_FILE,  X[TRAIN_COUNT + VALID_COUNT:])
    save_pkl(Y_TRAIN_FILE, labels[:TRAIN_COUNT])
    save_pkl(Y_VALID_FILE, labels[TRAIN_COUNT:(TRAIN_COUNT + VALID_COUNT)])
    save_pkl(Y_TEST_FILE,  labels[TRAIN_COUNT + VALID_COUNT:])
