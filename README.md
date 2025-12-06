
# About our Joke Model

Our joke model, trained on web scraped Reddit posts, attempts to output jokes based on user prompts.
Our intention was to train a model with an "edgy" feel that responds with natural and
contemporary sounding jokes. We built two models to implement this functionality. We
first built an LSTM RNN from scratch and trained the model on our datasets. This approach
was unsuccessfull, with our model unable to output coherent sounding sentences and
produced terrible perplexity and accuracy scores. Our second attempt involved 
fine tuning GPT2 via transfer learning techniques. This approach was much more successful,
producing much more reasonable objective metrics. The instructions below explain
the structure of our repository and the steps necessary to run our code.

# Repo Structure

## Datasets

Our datasets can be found in the compressed_data directory. After cloning/forking this
repository locally, your first step should be to create a new directory called
'data' and extract all contents in compressed_data to this new directory. We discuss
these datasets at length in our project report.

## Model Folders

The directories titled "model_3" and "lstm_rnn_models" contain our best
gpt2 model and lstm rnn model that we were able to train. DO NOT MODIFY THESE DIRECTORIES!
If you choose to retrain either model, the contents of these folders will be overwritten
after the model completely trains.

## Src Folder

This contains all our source code files. Instructions on how to run each file
can be found in the comments within each file itself. Here is a quick breakdown of each file:

### gpt2.py
Code to train/fine tune gpt-2 on our datasets via transfer learning techniques. Contains
functions to preprocess our data, fine tune our model, perform a grid search, evaluate
our model's performance, and generate jokes.

### lstm_rnn.py
Code to train an LSTM RNN from scratch on our dataset. Contains functions to
build, train, evaluate performance, and generate jokes.

### optimaljokeretrying.py
A script that generates a hundred jokes using our gpt2 model. This was used to generate
jokes for human evaluation. It is important to note that in order to filter out
harmful speech, there is a list of explicit words that could be triggering to some.

### prepare_data.py
Contains functions that help us preprocess our train, dev, and test data.

### toy_rnn.py
Our first attempt to create an RNN for our purposes. This is here only so you
can see our learning process, as we eventually moved onto a better code base.
This file is unused and is not meant to actually be run.
