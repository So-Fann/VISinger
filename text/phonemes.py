'''
This file extract the phonemes from the wesing opencpop dataset.
The phonemes extracted from this file is use for the symbols.py
'''
import os
import torch
from torch import nn
import sys
from pitch_id import pitch_id
from frame_prior_network import VariancePredictor
import mido
sys.path.append("../")

for _ in range(4):
	print(_)







# lf0={}
# for key in pitch_id:
# 	lf0[key] = torch.log(440 * (2 ** ((torch.tensor(pitch_id[key]) - 69) / 12))).item()
# print(lf0.items())
#
# lf0 = sorted(lf0.items(), key=lambda item: item[1])
# s_lf0 = []
# for key ,value in lf0:
# 	s_lf0.append(value)
# 	# str += str(value)
# 	# str+=", "
# print(s_lf0)
# print(str)

# gt_lf0 = torch.log(440 * (2 ** ((pitch_id - 69) / 12)))


# symbols = ['d', 'm', 'k', 's', 'ua', 'u', 'sh', 'ao', 'ch', 'ing', 'in', 'w', 'o', 'ia', 've', 'vn', 'n', 'y', 'z',
#             'uan', 'an', 'uai', 'uang', 'uo', 'er', 'iao', 'iong', 'ng', 'h', 'j', 'r', 'ui', 'en', 'f', 'c', 'ian',
#             'eng', 'iu', 'a', 'x', 'v', 'ai', 'ou', 'un', 'van', 'l', 'ong', 'ei', 'ie', 'b', 'e', 'g', 'p', 'q', 't',
#             'i', 'ang', 'zh', 'iang', 'SP', 'AP']
#
# list=[39, 58, 17, 42, 57, 50, 39, 10,  6, 55,  0, 50, 17, 55, 59, 57, 56, 59,
#          60, 59, 49, 25, 53,  9, 59, 60,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#           0,  0,  0,  0,  0]
# for num in list:
# 	print(symbols[num])

# phonemes_file = open("/Users/zhoushaohuan/Downloads/segments/phonemes.txt", "r")
#
# phonemes = []
#
# for line in phonemes_file.readlines():
# 	phoneme = line.replace("\n", "").split("|")[1]
# 	phoneme = phoneme.split(" ")
# 	phonemes.extend(phoneme)
# phonemes = set(phonemes)
# print(phonemes)
# print(len(phonemes))



