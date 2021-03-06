# Flow-sampleRNN

We try to improve the generation speed of SampleRNN model by using IAFs (inverse autoregressive flows).
This implementation is starting with a fork of [samplernn-pytorch](https://github.com/deepsound-project/samplernn-pytorch) implementation.

## Running the baseline model:
By default this program run flow-based model (`model_flow2.py`).
For running the baseline model (`model.py` has the implementation), we need to make following three changes : 
1. In `train.py` - replace `from model_flow2 import SampleRNN, Predictor` by `from model import SampleRNN, Predictor`
2. In `trainer/plugins.py` - replace `from model_flow2 import Generator` by `from model import Generator`
3. In `dataset.py` - replace `yield (sequences, reset, target_sequences)` by `yield (input_sequences, reset, target_sequences)`
Run as usual after making above three changes.

## Running on toy sine-waves
For training on this toy dataset use, `--dataset toy_sin_wave` argument while running `train.py`.
There is an implementation for generating the toy sine-waves in `sin_wave_data()` function in `dataset.py` file.
In `FolderDataset` class in same file, use the `toy_data_count` variable for defining the count of sequences in a single epoch and `toy_seq_len` variable for defining the length fo each sequence.

## New files added
* model_flow2.py
    - This file has the implementation of modified sampleRNN with IAF at sample-level.
    - Look at the diff of this file with model_flow2.py to see the changes made for implementing IAF at sample level.
* generate_from_model.py 
    - Meant for independently generating sequence using a trained model.
    - The code has the paths for the model and the model-parameters hard-coded in it.
    - This has not been tested on CPU. 
* test_generator.py
    - This is just for checking the time taken by the generator.
    - Again the model and its parameters are hard-coded as fo now.

Further details below in this README have been copied from the original forked repository.
![A visual representation of the SampleRNN architecture](http://deepsound.io/images/samplernn.png)

## Dependencies

This code requires Python 3.5+ and PyTorch 0.1.12+. Installation instructions for PyTorch are available on their website: http://pytorch.org/. You can install the rest of the dependencies by running `pip install -r requirements.txt`.

## Datasets

We provide a script for creating datasets from YouTube single-video mixes. It downloads a mix, converts it to wav and splits it into equal-length chunks. To run it you need youtube-dl (a recent version; the latest version from pip should be okay) and ffmpeg. To create an example dataset - 4 hours of piano music split into 8 second chunks, run:

```
cd datasets
./download-from-youtube.sh "https://www.youtube.com/watch?v=EhO_MrRfftU" 8 piano
```

You can also prepare a dataset yourself. It should be a directory in `datasets/` filled with equal-length wav files. Or you can create your own dataset format by subclassing `torch.utils.data.Dataset`. It's easy, take a look at `dataset.FolderDataset` in this repo for an example.

## Training

To train the model you need to run `train.py`. All model hyperparameters are settable in the command line. Most hyperparameters have sensible default values, so you don't need to provide all of them. Run `python train.py -h` for details. To train on the `piano` dataset using the best hyperparameters we've found, run:

```
python train.py --exp TEST --frame_sizes 4 16 --n_rnn 2 --dataset piano --keep_old_checkpoints
```

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.

We also have an option to monitor the metrics using [CometML](https://www.comet.ml/). To use it, just pass your API key as `--comet_key` parameter to `train.py`.
