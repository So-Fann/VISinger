# -*- coding: utf-8 -*-
import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import phonemes_to_sequence
from text.pitch_id import pitch_id


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        # 从training_files中加载的音频地址以及音频内容
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate
        self.wavs_path      = hparams.wavs_path

        # self.add_blank = hparams.add_blank
        # self.min_text_len = getattr(hparams, "min_text_len", 1)
        # self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        lengths = []
        audiopath_and_text_new = []
        # 取出每一行的音频地址audiopath和音频内容text

        for audiopath, text, phonemes, note_pitch, note_dur, phn_dur, slur_flag in self.audiopaths_and_text:
            # get_size获取文件大小（字节数），这里计算wav的长度，根据上方计算公式得出结果
            wav_path = os.path.join(self.wavs_path, audiopath)+".wav"
            lengths.append(os.path.getsize(wav_path) // (2 * self.hop_length))
            audiopath_and_text_new.append([wav_path, text, phonemes, note_pitch, note_dur, phn_dur, slur_flag])
        self.lengths = lengths
        self.audiopaths_and_text = audiopath_and_text_new


    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text

        audiopath,  phonemes, note_pitch, note_dur, phn_dur, slur_flag = \
            audiopath_and_text[0], audiopath_and_text[2], audiopath_and_text[3],\
            audiopath_and_text[4], audiopath_and_text[5], audiopath_and_text[6]

        phonemes = self.get_phonemes(phonemes)
        note_pitch = self.get_pitchid(note_pitch)
        note_dur, phn_dur, slur_flag = self.get_duration_flag(note_dur, phn_dur, slur_flag)



        # 得到文本内容、频谱图、音频数据
        spec, wav = self.get_audio(audiopath)
        return phonemes, note_pitch, note_dur, phn_dur, slur_flag, spec, wav

    def get_phonemes(self, phonemes):
        phonemes_norm = phonemes_to_sequence(phonemes)
        phonemes_norm = torch.LongTensor(phonemes_norm)
        return phonemes_norm

    def get_pitchid(self, note_pitch):
        note_pitch_new = []
        note_pitch = note_pitch.split(" ")
        for note in note_pitch:
            note = note.split("/")[0]
            pitch = pitch_id[note]
            note_pitch_new.append(pitch)
        return torch.FloatTensor(np.array(note_pitch_new).astype(np.float32))

    def get_duration_flag(self, note_dur, phn_dur, slur_flag):
        note_dur = note_dur.split(" ")
        phn_dur = phn_dur.split(" ")
        slur_flag = slur_flag.split(" ")
        note_dur = torch.FloatTensor(np.array(note_dur).astype(np.float32))
        phn_dur = torch.FloatTensor(np.array(phn_dur).astype(np.float32))
        slur_flag = torch.FloatTensor(np.array(slur_flag).astype(np.float32))
        return note_dur, phn_dur, slur_flag


    def get_audio(self, filename):
        # 使用scipy.io.wavfile.read读取的音频文件
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            fsize = os.path.getsize(spec_filename)
            if fsize == 0:
                print("spec_filesize: ", str(fsize)+"  |  "+spec_filename)
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[6].size(1) for x in batch]),
            dim=0, descending=True)
        # spec和wav都是tensor，所以可以用size取大小。其余字符串得用len函数取大小


        max_phonemes_len = max([len(x[0]) for x in batch])
        max_notepitch_len = max([len(x[1]) for x in batch])
        max_notedur_len = max([len(x[2]) for x in batch])
        max_phndur_len = max([len(x[3]) for x in batch])
        max_slurflag_len = max([len(x[4]) for x in batch])
        max_spec_len = max([x[5].size(1) for x in batch])
        max_wav_len = max([x[6].size(1) for x in batch])


        phonemes_lengths = torch.LongTensor(len(batch))
        notepitch_lengths = torch.LongTensor(len(batch))
        notedur_lengths = torch.LongTensor(len(batch))
        phndur_lengths = torch.LongTensor(len(batch))
        slurflag_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))


        phonemes_padded = torch.LongTensor(len(batch), max_phonemes_len)
        notepitch_padded = torch.LongTensor(len(batch), max_notepitch_len)
        notedur_padded = torch.FloatTensor(len(batch), max_notedur_len)
        phndur_padded = torch.FloatTensor(len(batch), max_phndur_len)
        slurflag_padded = torch.LongTensor(len(batch), max_slurflag_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][5].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)


        phonemes_padded.zero_()
        notepitch_padded.zero_()
        notedur_padded.zero_()
        phndur_padded.zero_()
        slurflag_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            phonemes = row[0]

            phonemes_padded[i, :phonemes.size(0)] = phonemes
            phonemes_lengths[i] = phonemes.size(0)

            notepitch = row[1]

            notepitch_padded[i, :notepitch.size(0)] = notepitch
            notepitch_lengths[i] = notepitch.size(0)

            notedur = row[2]

            notedur_padded[i, :notedur.size(0)] = notedur
            notedur_lengths[i] = notedur.size(0)

            phndur = row[3]
            phndur_padded[i, :phndur.size(0)] = phndur
            phndur_lengths[i] = phndur.size(0)

            slurflag = row[4]
            slurflag_padded[i, :slurflag.size(0)] = slurflag
            slurflag_lengths[i] = slurflag.size(0)

            spec = row[5]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[6]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return phonemes_padded, phonemes_lengths,\
                   notepitch_padded, notepitch_lengths,\
                   notedur_padded, notedur_lengths,\
                   phndur_padded, phndur_lengths,\
                   slurflag_padded, slurflag_lengths,\
                   spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing

        return  phonemes_padded, phonemes_lengths,\
                   notepitch_padded, notepitch_lengths,\
                   notedur_padded, notedur_lengths,\
                   phndur_padded, phndur_lengths,\
                   slurflag_padded, slurflag_lengths,\
                   spec_padded, spec_lengths, wav_padded, wav_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        # buckets = [[],[],[],[],[],...]
        buckets = [[] for _ in range(len(self.boundaries) - 1)]

        for i in range(len(self.lengths)):
            length = self.lengths[i]

            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
