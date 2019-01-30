# generate music using a particular model

from model import SampleRNN, Predictor, Generator

import os, sys, time

import torch
from librosa.output import write_wav


# edit the modeldir and other variables below
modeldir = "colab_results/"
modelname = "ep1-it625"
audio_pref = "r001_{}"
save_raw = False
n_samples = 1
sample_length = 800
sample_rate = 16000


samples_path = os.path.join(modeldir, "gen_samples")
os.makedirs(samples_path, exist_ok=True)

# sys.stderr.write("available models are: {}".format(listdir(modeldir)))
modelpath = os.path.join(modeldir, modelname)

srnn_model1 = SampleRNN(frame_sizes=[4, 16], n_rnn=2, dim=1024, learn_h0=True, 
                    q_levels=256, weight_norm=True)

if torch.cuda.is_available():
    srnn_model1 = srnn_model1.cuda()

predictor1 = Predictor(srnn_model1)

if torch.cuda.is_available():
    predictor1 = predictor1.cuda()

if torch.cuda.is_available():
    predictor1.load_state_dict(torch.load(modelpath)['model'])
else:
    predictor1.load_state_dict(torch.load(modelpath, map_location='cpu')['model'])

print("model loaded successfully!")

generate = Generator(srnn_model1, True)

import time
s_time = time.time()

sys.stderr.write("Generating {} sequences, each of length {}."\
                .format(n_samples, sample_length))
samples = generate(n_samples, sample_length).cpu().float().numpy()

sys.stderr.write("Total time taken = {}".format(time.time() - s_time))

for i in range(n_samples):
    if save_raw:
        samples.tofile('debug_seq_{}.csv'.format(i),
                        sep=',', format='%10.5f')
    write_wav(
        os.path.join(
            samples_path, audio_pref.format(i + 1)
        ),
        samples[i, :], sr=sample_rate, norm=True
    )
