import time
import numpy as np
from model_flow2 import SampleRNN, Generator
# from model import SampleRNN, Generator


model1 = SampleRNN(frame_sizes=[4, 16], n_rnn=2, dim=1024, learn_h0=True, 
                    q_levels=256, weight_norm=True)

generator1 = Generator(model1)

n_seqs, seq_len = (1, 1600)

s_time = time.time()

rand_samples = generator1(n_seqs = n_seqs, seq_len = seq_len)

print(rand_samples.flatten())
print(rand_samples.shape)
print("Time taken for generating {} sequences of {} length = {}"\
            .format(n_seqs, seq_len, time.time() - s_time))
