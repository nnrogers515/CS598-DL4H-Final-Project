# CS598-DL4H-Final-Project

## Code and Project Description:
The code in this repository is either re-used or heavily based off the code from https://github.com/danicaxiao/CONTENT. A repository which was created to house the CONTENT model utilized in a paper for detecting hospital readmission through deep contextual embedding found here: https://doi.org/10.1371/journal.pone.0195024

## How to Use

1. Clone this repository locally using the method of your choice. For the simplest to setup, cloning over http with standard github credentials would be best. The command for this you would type into your terminal that has access to the `git` CLI tool: `git clone https://github.com/nnrogers515/CS598-DL4H-Final-Project.git`
2. Before running any code you will need to use `cd CS598-DL4H-Final-Project` to enter the project directory then install the necessary imports via `pip install -r requirements.txt`
3. After cloning the repository and installing the dependencies, you can simply run `make run` and the model should run it's classification based off of the data sample provided in `data/S1_File.txt` you can replace this file with your own file given it matches the same csv formatting as this file. For convenience, the data folder will be created with already pre-processed files but S1_File.txt will be repulled through the code if needed as it is quite large.
4. If you wish to make any changes to how the model is trained you can edit the `Configuration.py` file and use `make train`.

NOTE: If you are unable to use `make` then use `python3 Main.py train` for training or `python3 Main.py run` for just testing.

## Code Flow

1. We start in `DataPrep.py` where we pull the data, pre-process the data and set up the splits for training, validation, and testing.
2. Next this data is read in through `PatientDataLoader.py` along with the information in `Configuration.py` to produce the files utilized by `CONTENT.py` and `CONTENT_FixedBatch.py` where the model is trained and produces it's classification predictions on whether a patient is likely to be readmitted to the hospital (1) or not (0).


# Sources

## Paper Citation
Xiao C, Ma T, Dieng AB, Blei DM, Wang F (2018) Readmission prediction via deep contextual embedding of clinical concepts. PLOS ONE 13(4): e0195024. https://doi.org/10.1371/journal.pone.0195024

## Code Base Citation

Xiao, C., Ma, T., Dieng, A., Blei, D., & Wang, F. (2017). CONTENT (Version 1.0.0) [Computer software]. https://doi.org/10.1371/journal.pone.0195024