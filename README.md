# CS598-DL4H-Final-Project

## Code and Project Description:
The code in this repository is either re-used or heavily based off the code from https://github.com/danicaxiao/CONTENT. A repository which was created to house the CONTENT model utilized in a paper for detecting hospital readmission through deep contextual embedding found here: https://doi.org/10.1371/journal.pone.0195024

## How to Use

1. Clone this repository locally using the method of your choice. For the simplest to setup, cloning over http with standard github credentials would be best. The command for this you would type into your terminal that has access to the `git` CLI tool: `git clone https://github.com/nnrogers515/CS598-DL4H-Final-Project.git`
2. Before running any code you will need to use `cd CS598-DL4H-Final-Project` to enter the project directory then install the necessary imports via `pip install -r requirements.txt`
3. Due to additional import complications, you will also need to manually upgrade versions for Theano and Lasagne using the follow commands:
   1. `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`
4. On the first run, it is best to use `make new` which should provide a setup for the input dataset provided or download the data directly if no dataset at the path `data/S1_File.txt` exists.
5. For further runs, you can simply run `make run` and the model should run it's classification based off of the data sample provided in `data/S1_File.txt` you can replace this file with your own file given it matches the same csv formatting as this file. For convenience, the data folder will be created with already pre-processed files but S1_File.txt will be repulled through the code if needed as it is quite large.
6. If you wish to make any changes to how the model is trained you can edit the `Configuration.py` file and use `make new` again, the splits for the data can be adjusted at the top of `DataPrep.py` should you have new data with different split dimensions.

NOTE: If you are unable to use `make` then use `python3 Main.py new` for a new run or `python3 Main.py` instead of `make run`. There is also the fixed batch size version of the code which can be ran through `python3 Main.py new_fixed` or `python3 Main.py fixed` but this is still a work-in-progress so ignore this for now.

## Code Flow

1. Data Pre-Processing Code can be found in `DataPrep.py`
2. Training and Evaluation Code are found in either `CONTENT.py` or `CONTENT_FixedBatch.py`
3. The Pretrained model will be stored in the `model/` folder in this repository

## Results Table

Data for Run #1 Is Tentative as kinks are being worked out

| Experiment Run      |  #1 (Tmp)  |     #2     |     #3     |     #4     |     #5     |     #6     |     #7     |     #8     |     #9     |     #10    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ROC-AUC             | 0.7956     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     |
| Accuracy(%)         | 83.91%     | 0.0000%    | 0.0000%    | 0.0000%    | 0.0000%    | 0.0000%    | 0.0000%    | 0.0000%    | 0.0000%    | 0.0000%    |
| Precision           | 0.7493     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     |
| Recall              | 0.4091     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     |
| F1-Score            | 0.5264     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     |
| CPU/GPU Hours       | 1.61       | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     |
| Memory Usage (GB)   | 145.62     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 0.0000     |

- Average ROC-AUC Achieved: 0.0000 ± 0.0000
- Average Accuracy Achieved: 0.0000% ± 0.0000%
- Average Precision Achieved: 0.0000 ± 0.0000
- Average Recall Achieved: 0.0000 ± 0.0000
- Average F1-Score Achieved: 0.0000 ± 0.0000
- Average CPU/GPU Hours: 0.0000 ± 0.0000 Hours
- Average Memory Usage: 0.0000 ± 0.0000 GB
## Dependencies

The dependencies required for this project can be found inside of the requirements.txt and can be downloaded via `pip install -r requirements.txt` or via the `conda` CLI tool as well, but here are the dependencies listed out for convenience:

- urllib3
- pandas
- numpy
- zipfile36
- matplotlib
- theano
- lasagne
- sklearn
- psutil

NOTE: You will need to download Lasagne 0.2.dev1 to work with Theano version 1.0.5. To do so, run the following commands in your python environment:

```
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```


## Datasets and Downloads

The dataset for this should be downloaded through the code pipeline, but in the case of there being any issues, you can use the `txtData.zip` file in this repository and extract this into a `data/` folder. This zip file can be downloaded directly from the paper at https://doi.org/10.1371/journal.pone.0195024 via this link https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0195024.s001

# Sources

## Paper Citation
Xiao C, Ma T, Dieng AB, Blei DM, Wang F (2018) Readmission prediction via deep contextual embedding of clinical concepts. PLOS ONE 13(4): e0195024. https://doi.org/10.1371/journal.pone.0195024

## Code Base Citation

Xiao, C., Ma, T., Dieng, A., Blei, D., & Wang, F. (2017). CONTENT (Version 1.0.0) [Computer software]. https://doi.org/10.1371/journal.pone.0195024
