import os
import re
import time
import hashlib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.io import wavfile as wf
import librosa
from tqdm import tqdm


wanted_words = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go",
                "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven",
                "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"]

wanted_words_to_class = {}
for i in range(len(wanted_words)):
    wanted_words_to_class[wanted_words[i]] = i


MAX_NUM_WAVES_PER_CLASS = 2**27 - 1  # ~134M


def which_set(filename, testing_percentage=20):
    """Determines which data partition the file should belong to. (Copied from README)"""
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVES_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVES_PER_CLASS))
    if percentage_hash < testing_percentage:
        result = 'test'
    else:
        result = 'training'
    return result


def pre_processing_dataset(data_path='./data_speech_commands_v0.02/', transform=None):
    columns = ['wav', 'label_str']
    train_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
    # Build the wav path
    for x in os.listdir(data_path):
        if os.path.isdir(data_path + '/' + x) and x in wanted_words:
            print('Handle class: <'+x+'>.')
            all_files = [y for y in os.listdir(data_path + x) if '.wav' in y]
            for file in all_files:
                full_path = data_path + x + '/' + file
                sample_rate, sample = wf.read(full_path)
                if sample.shape[0] != 16000:
                    continue
                split_set = which_set(file)
                if split_set == 'test':
                    test_df = test_df.append({'wav': full_path, 'label_str': x}, ignore_index=True)
                else:
                    train_df = train_df.append({'wav': full_path, 'label_str': x}, ignore_index=True)
    train_df.to_csv(path_or_buf=data_path+'training_set.csv', index=False)
    test_df.to_csv(path_or_buf=data_path+'test_set.csv', index=False)
    train_buffer_x = []
    train_buffer_y = []
    for index in tqdm(range(len(train_df))):
        wav_name = train_df.iloc[index, 0]
        wav_sample_rate, wav = wf.read(wav_name)
        sample = GoogleSpeechDataset.mel_log(wav, wav_sample_rate).astype('float')
        if transform:
            sample = transform(sample)
        train_buffer_x.append(sample)
        train_buffer_y.append(wanted_words_to_class[train_df.iloc[index, 1]])
    train_x = np.concatenate(train_buffer_x, axis=0)
    train_y = np.array(train_buffer_y)
    np.save(data_path+'train_x.npy', train_x)
    np.save(data_path+'train_y.npy', train_y)
    test_buffer_x = []
    test_buffer_y = []
    for index in tqdm(range(len(test_df))):
        wav_name = test_df.iloc[index, 0]
        wav_sample_rate, wav = wf.read(wav_name)
        sample = GoogleSpeechDataset.mel_log(wav, wav_sample_rate).astype('float')
        if transform:
            sample = transform(sample)
        test_buffer_x.append(sample)
        test_buffer_y.append(wanted_words_to_class[test_df.iloc[index, 1]])
    test_x = np.concatenate(test_buffer_x, axis=0)
    test_y = np.array(test_buffer_y)
    np.save(data_path+'test_x.npy', test_x)
    np.save(data_path+'test_y.npy', test_y)


class GoogleSpeechDataset(Dataset):
    def __init__(self, input_file, root_dir, transform=None, load_cached=True):
        self.root_dir = root_dir
        self.transform = transform
        self.load_cached = load_cached
        print('Pin google speech recognition dataset in the RAM.')
        start_time = time.time()
        if load_cached:
            self.buffer_x = np.load(input_file+'_x.npy')
            self.buffer_y = np.load(input_file + '_y.npy')
        else:
            self.wav_frame = pd.read_csv(input_file)
            self.buffer = []
            for index in tqdm(range(len(self.wav_frame))):
                wav_name = os.path.join(self.root_dir, self.wav_frame.iloc[index, 0])
                wav_sample_rate, wav = wf.read(wav_name)
                sample = {'wav': GoogleSpeechDataset.mel_log(wav, wav_sample_rate).astype('float'),
                          'label': wanted_words_to_class[self.wav_frame.iloc[index, 1]]}
                if self.transform:
                    sample = self.transform(sample)
                self.buffer.append(sample)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Data loading is done. Takes {:3.2f}s.'.format(elapsed_time))

    def __len__(self):
        if self.load_cached:
            return len(self.buffer_x)
        else:
            return len(self.buffer)

    def __getitem__(self, index):
        if self.load_cached:
            return {'wav': self.buffer_x[i], 'label': self.buffer_y[i]}
        else:
            return self.buffer[index]

    @staticmethod
    def mel_log(audio, sample_rate):
        s = librosa.feature.melspectrogram(audio.astype(np.float), sr=sample_rate, n_mels=128)
        log_s = librosa.power_to_db(s, ref=np.max).reshape(-1)
        return log_s


def train_dataset():
    return GoogleSpeechDataset(input_file='./data_speech_commands_v0.02/train', root_dir='.', load_cached=True)


def test_dataset():
    return GoogleSpeechDataset(input_file='./data_speech_commands_v0.02/test', root_dir='.', load_cached=True)


def main():
    pre_processing_dataset()


if __name__ == '__main__':
    main()
