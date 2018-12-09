import os
import numpy as np
from librosa.output import write_wav
import time

def get_batch(batch_size, timesteps):
    low_freq_factor = np.random.uniform(size=(batch_size,))
    high_freq_factor = np.random.uniform(size=(batch_size,))

    x = np.arange(0, np.pi, np.pi/timesteps)
    low_y = (low_freq_factor + 1)[:, None]*x[None, :]
    high_y = 20.*(high_freq_factor + 1)[:, None]*x[None, :]

    noise = np.random.uniform(low=-0.001, high=0.001, size=high_y.shape)
    batch = np.sin(high_y)*np.sin(low_y) + noise
    return batch.astype('float32')


def create_sin_wave_data(data_dir, num_files, seq_len, sample_rate=16000, save_raw=False):

    os.makedirs(data_dir, exist_ok=True)
    dataset = get_batch(num_files, seq_len)
    for i in range(num_files):
        if save_raw:
            dataset[i].tofile('sin_{}.csv'.format(i),
                            sep=',', format='%7.5f')
        write_wav(
            os.path.join(
                data_dir, 'sin_{}.wav'.format(i)
            ),
            dataset[i, :], sr=sample_rate, norm=False
        )


if __name__ == "__main__":

    data_dir = "datasets/sin_wave/"
    file_len = 16000 * 8
    num_files = 512

    begin = time.time()

    create_sin_wave_data(data_dir, num_files, file_len)
    
    print("Total time taken for generating {} sequences of {} length = {}"\
            .format(num_files, file_len, time.time() - begin))
