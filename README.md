# CS598-DL4H-Final-Project

## Code and Project Description:
The code in this repository is either re-used or heavily based off the code from https://github.com/danicaxiao/CONTENT. A repository which was created to house the CONTENT model utilized in a paper for detecting hospital readmission through deep contextual embedding found here: https://doi.org/10.1371/journal.pone.0195024

## How to Use

1. Clone this repository locally using the method of your choice. For the simplest to setup, cloning over http with standard github credentials would be best. The command for this you would type into your terminal that has access to the `git` CLI tool: `git clone https://github.com/nnrogers515/CS598-DL4H-Final-Project.git`
2. Before running any code you will need to use `cd CS598-DL4H-Final-Project` to enter the project directory then install the necessary imports via `pip install -r requirements.txt`
   1. See Dependencies Section of this README.md if you are having more trouble!
3. Due to additional import complications, you will also need to manually upgrade versions for Lasagne using the follow commands (Don't worry if it says it failed to install the wheel, as long as it says `Successfully installed Lasagne-0.2.dev1` and `Successfully installed Theano-1.0.5+unknown` you should be fine):
   1. `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`
   2. `pip install --upgrade https://github.com/Theano/Theano/archive/master.zip`
4. On the first setup, it is best to use `make new` which should provide a setup for the input dataset provided or download the data directly if no dataset at the path `data/S1_File.txt` exists.
5. Then for a general run with training and testing you can simply use `make train_continued` and the pre-trained model should be trained and then you can test on its classification using `make test` (you can run `make test` right away if you don't want to wait til training finishes), these use data based off of the data sample provided in `data/S1_File.txt` you can replace this file with your own file given it matches the same csv formatting as this file, if you do this, makes sure to use `make new`. For convenience, the data folder will be created with already pre-processed files but `S1_File.txt` will be re-pulled through the code if needed as it is quite large.
6. If you wish to make any changes to how the model is trained you can edit the `Configuration.py` file. The splits for the data can be adjusted at the top of `DataPrep.py` should you have new data that you want to split a certain way.
7. To train an entirely new model just use `make train` and the pre-trained model will be overwritten.

NOTE: If you are unable to use `make` then use `python3 Main.py new` for a new run or `python3 Main.py train continued` instead of `make train_continued`. If you want to see what commands should replace the make commands look at the contents of the `Makefile` and look at what commands are run for each section.

For more specific ways to train, test, and evaluate see the sections below. Make sure you use `make new` before using them!
## How to Train a New Model

Make sure you have already run `make new`:

In the Project Directory after all dependencies have been installed, run the command:

```bash
make train
```

or if you don't have `make`, run the command:

```bash
python3 main.py train
```
This will train a new model and run it against validation data, then save the model locally as `model.npz`

## How to Train a Saved Model

Make sure you have already run `make new`:

In the Project Directory after all dependencies have been installed, run the command:

```bash
make train_continued
```

or if you don't have `make`, run the command:

```bash
python3 Main.py train continued
```
This will train a saved model under the name `model.npz` that exists in the current working directory, 
run against validation data and then resave the model to the same file.
## How to Test

Make sure you have already run `make new`:

In the Project Directory after all dependencies have been installed, run the command:

```bash
make test
```

or if you don't have `make`, run the command:

```bash
python3 Main.py test
```

This will test the trained model (named `model.npz`) in the project directory against the dataset loaded from `data/S1_File.txt`
Note: If the pretrained model isn't available you will have to train one first

## How to Evaluate

Make sure you have already run `make new`:

In the Project Directory after all dependencies have been installed, run the command:

```bash
make eval
```

or if you don't have `make`, run the command:

```bash
python3 Main.py eval
```

This will evaluate the model results as done in the original project and show a plot of precision vs recall for the test results.
This uses output from the training and testing runs rather than the model itself, so note you will have to run training and testing for this to update
## Code Flow

1. `Main.py` is the entry point that calls all other methods and sets up the project
2. Data Pre-Processing Code can be found in `DataPrep.py` which pulls from and also fills-in the `data` folder.
3. Training, Testing and Evaluation Code are found in either `CONTENT.py` and utilize `Configuration.py` and `PatientDataLoader.py`
4. The Pretrained model from training will be stored as `model.npz` in this directory.

## Results Table

Data for 10 Experiment Runs:

| Experiment Run      |  #1 (Tmp)  |     #2     |     #3     |     #4     |     #5     |     #6     |     #7     |     #8     |     #9     |     #10    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ROC-AUC             | 0.7935     | 0.8006     | 0.8009     | 0.7998     | 0.7995     | 0.7973     | 0.7931     | 0.7905     | 0.7914     | 0.7888     |
| Accuracy(%)         | 83.05%     | 83.61%     | 83.82%     | 83.72%     | 84.08%     | 84.01%     | 83.81%     | 83.99%     | 83.83%     | 83.63%     |
| Precision           | 0.7074     | 0.7526     | 0.7572     | 0.6940     | 0.7251     | 0.7202     | 0.7092     | 0.7741     | 0.7370     | 0.6991     |
| Recall              | 0.3901     | 0.3791     | 0.3882     | 0.4639     | 0.4444     | 0.4454     | 0.4464     | 0.3836     | 0.4108     | 0.4482     |
| F1-Score            | 0.5029     | 0.5042     | 0.5133     | 0.5561     | 0.5511     | 0.5504     | 0.5479     | 0.5130     | 0.5275     | 0.5462     |
| CPU/GPU Hours       | 1.30       | 1.46       | 1.24       | 1.26       | 1.25       | 1.33       | 1.29       | 1.24       | 1.25       | 1.30       |
| Memory Usage (GB)   | 521.56     | 526.03     | 407.78     | 408.86     | 411.11     | 410.58     | 414.10     | 413.39     | 416.76     | 416.72     |

- Average ROC-AUC Achieved: 0.7955 ± 0.0046
- Average Accuracy Achieved: 83.76% ± 0.2932%
- Average Precision Achieved: 0.7276 ± 0.0269
- Average Recall Achieved: 0.4200 ± 0.0327
- Average F1-Score Achieved: 0.5313 ± 0.0213
- Average CPU/GPU Hours: 1.29 ± 0.066 Hours
- Average Memory Usage: 434.69 ± 47.068 GB

## Dependencies

The dependencies required for this project can be found inside of the requirements.txt and can be downloaded via `pip install -r requirements.txt` or via the `conda` CLI tool as well, but here are the dependencies listed out for convenience:

- urllib3
- pandas
- numpy===1.20.3 (Downgraded Version Needed for Theano and Lasagne)
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

If you are still having issues running the app, create a virtual python environment using `python3 -m venv .` inside of the cloned github repos, then use `source bin/activate`, then explicitly download the versions and dependencies listed in the `dependencyConfirmation.txt` file. This is the Nuclear option, but if all else fails this environment setup should work no matter what!

## Datasets and Downloads

The dataset for this should be downloaded through the code pipeline, but in the case of there being any issues, you can use the `txtData.zip` file in this repository and extract this into a `data/` folder. This zip file can be downloaded directly from the paper at https://doi.org/10.1371/journal.pone.0195024 via this link https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0195024.s001

## Additional Info

Communication with the Author of the Paper Has Been Added as [author_communication.PNG](https://github.com/nnrogers515/CS598-DL4H-Final-Project/blob/main/author_communication.PNG) if needed.

# Sources

## Paper Citation
Xiao C, Ma T, Dieng AB, Blei DM, Wang F (2018) Readmission prediction via deep contextual embedding of clinical concepts. PLOS ONE 13(4): e0195024. https://doi.org/10.1371/journal.pone.0195024

## Code Base Citation

Xiao, C., Ma, T., Dieng, A., Blei, D., & Wang, F. (2017). CONTENT (Version 1.0.0) [Computer software]. https://doi.org/10.1371/journal.pone.0195024
