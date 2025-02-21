
from funasr import AutoModel
import torch
import soundfile as sf
import torchaudio as ta
from math import ceil
import math
import numpy as np


folder = 'eg2/'
asr_file = 'buffer_eg_v2.txt'
#------------ Load data------------
waveform, sample_rate = sf.read(folder + "eg16k.wav")
print(waveform.shape)
if waveform.shape[1] > 1:
    waveform = np.expand_dims(waveform[:, 0], axis=1)
stride = sample_rate * 2
piece_num = ceil(len(waveform) / stride)

#------------ load speaker Verification model------------
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import os
import sys
import re
import pathlib
import argparse

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download

parser = argparse.ArgumentParser(description='Extract speaker embeddings.')
parser.add_argument('--model_id', default='iic/speech_campplus_sv_zh-cn_16k-common', type=str, help='Model id in modelscope')
parser.add_argument('--wavs', nargs='+', type=str, help='Wavs')
parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2NetV2_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 26,
        'scale': 2,
        'expansion': 2,
    },
}

ERes2NetV2_w24s4ep4_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 24,
        'scale': 4,
        'expansion': 4,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

ECAPA_CNCeleb = {
    'obj': 'speakerlab.models.ecapa_tdnn.ECAPA_TDNN.ECAPA_TDNN',
    'args': {
        'input_size': 80,
        'lin_neurons': 192,
        'channels': [1024, 1024, 1024, 1024, 3072],
    },
}

supports = {
    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    # ERes2Net trained on 200k labeled speakers
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    # ERes2NetV2 trained on 200k labeled speakers
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_COMMON,
        'model_pt': 'pretrained_eres2netv2.ckpt',
    },
    # ERes2NetV2_w24s4ep4 trained on 200k labeled speakers
    'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_w24s4ep4_COMMON,
        'model_pt': 'pretrained_eres2netv2w24s4ep4.ckpt',
    },
    # ERes2Net_Base trained on 200k labeled speakers
    'iic/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
    },
    # CAM++ trained on a large-scale Chinese-English corpus
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
    # CAM++ trained on VoxCeleb
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    # ERes2Net trained on VoxCeleb
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    # ERes2Net_Base trained on 3dspeaker
    'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1', 
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    # ERes2Net_large trained on 3dspeaker
    'iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
    # ECAPA-TDNN trained on CNCeleb
    'iic/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on 3dspeaker
    'iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on VoxCeleb
    'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k': {
        'revision': 'v1.0.1', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa_tdnn.bin',
    },
}

args = parser.parse_args()
if args.model_id.startswith('damo/'):
    args.model_id = args.model_id.replace('damo/','iic/', 1)
assert args.model_id in supports, "Model id not currently supported."
save_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1])
save_dir = pathlib.Path(save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

conf = supports[args.model_id]
# download models from modelscope according to model_id
cache_dir = snapshot_download(
            args.model_id,
            revision=conf['revision'],
            )
cache_dir = pathlib.Path(cache_dir)

embedding_dir = save_dir / 'embeddings'
embedding_dir.mkdir(exist_ok=True, parents=True)

# link
download_files = ['examples', conf['model_pt']]
for src in cache_dir.glob('*'):
    if re.search('|'.join(download_files), src.name):
        dst = save_dir / src.name
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        dst.symlink_to(src)

pretrained_model = save_dir / conf['model_pt']
pretrained_state = torch.load(pretrained_model, map_location='cpu')

if torch.cuda.is_available():
    msg = 'Using gpu for inference.'
    print(f'[INFO]: {msg}')
    device = torch.device('cuda')
else:
    msg = 'No cuda device is detected. Using cpu.'
    print(f'[INFO]: {msg}')
    device = torch.device('cpu')

# load model
model = conf['model']
embedding_model = dynamic_import(model['obj'])(**model['args'])
embedding_model.load_state_dict(pretrained_state)
#embedding_model.to(device)
embedding_model.eval()

feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
def compute_embedding(wav):
    # compute feat
    feat = feature_extractor(wav).unsqueeze(0)#.to(device)
    # compute embedding
    with torch.no_grad():
        embedding = embedding_model(feat.float()).detach().squeeze(0).cpu().numpy()
    return embedding
def compute_score(embedding1, embedding2):
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    scores = similarity(torch.from_numpy(embedding1).unsqueeze(0), torch.from_numpy(embedding2).unsqueeze(0)).item()
    return scores


# ------------------load ASR model----------------
model = AutoModel(model="paraformer-zh",  
                  vad_model="fsmn-vad", 
                  punc_model="ct-punc", disable_update=True,
                  spk_model="cam++"
                  )
    
# -------------------computing...----------------
low_threshold = 0.6
high_threshold = 0.84
max_buffer_length = 20 * sample_rate
FRAME_PER_SECOND = 1000

buffer = None

spks = [] # shape=[spk num, emb length]
spks_num = []
last_id = -1
now_id = -1
stage_id = -1
stage_piece = None
for i in range(piece_num):
    if i != piece_num - 1:
        sf.write(folder + "piece.wav", waveform[i*stride: (i+1)*stride],sample_rate)
    else:
        sf.write(folder + "piece.wav", waveform[i*stride: ],sample_rate)

    piece, sample_rate = sf.read(folder + "piece.wav")
    emb = compute_embedding(torch.from_numpy(piece).unsqueeze(0))

    # update spks and get spk id of this piece
    score = []

    for spk in spks:
        score.append(compute_score(emb, spk))

    if len(spks) == 0:
        # first piece
        spks.append(emb)
        spks_num.append(1)
        now_id = 0
        last_id = 0
    else:
        # not first piece
        if min(score) < high_threshold:
            now_id = id = np.argmin(np.array(score))
            spks[id] = (spks[id]*spks_num[id] + emb) / (spks_num[id] + 1)
            spks_num[id] += 1
        else:
            spks.append(emb)
            spks_num.append(1)
            now_id = len(spks) - 1

        if stage_id != -1:
            if last_id == now_id:
                # spk change
                sf.write(folder + "buffer.wav", buffer, sample_rate)
                buffer = np.concatenate((stage_piece, piece), axis = 0)
                asr_result = model.generate(input=folder + "buffer.wav", 
                            batch_size_s=6000, 
                            hotword='增容'
                            )
                with open(folder + asr_file,'a') as f:
                    f.write(str(stage_id) + asr_result[0]['text'] + '\n')   
            else:
                if now_id == stage_id:
                    # spk not change
                    buffer = np.concatenate((buffer, stage_piece), axis = 0)
                    buffer = np.concatenate((buffer, piece), axis = 0)
                else:
                    # spk change
                    sf.write(folder + "buffer.wav", buffer, sample_rate)
                    asr_result = model.generate(input=folder + "buffer.wav", 
                                batch_size_s=6000, 
                                hotword='增容'
                                )
                    with open(folder + asr_file,'a') as f:
                        f.write(str(stage_id) + asr_result[0]['text'] + '\n')   

                    sf.write(folder + "buffer.wav", stage_piece, sample_rate)
                    asr_result = model.generate(input=folder + "buffer.wav", 
                                batch_size_s=6000, 
                                hotword='增容'
                                )
                    with open(folder + asr_file,'a') as f:
                        f.write(str(last_id) + asr_result[0]['text'] + '\n')   
                    
                    buffer = piece

            stage_id = -1
            stage_piece = None
        elif buffer is not None and last_id != now_id:
            if min(score) < low_threshold:
                # confidant
                sf.write(folder + "buffer.wav", buffer, sample_rate)
                buffer = piece

                asr_result = model.generate(input=folder + "buffer.wav", 
                            batch_size_s=6000, 
                            hotword='增容'
                            )
                with open(folder + asr_file,'a') as f:
                    f.write(str(last_id) + asr_result[0]['text'] + '\n')  
            else:
                # not confidant
                stage_id = last_id
                stage_piece = piece
        elif buffer is not None and len(buffer) > max_buffer_length:
            # go back to update ASR
            buffer = np.concatenate((buffer, piece), axis = 0)
            sf.write(folder + "buffer.wav", buffer, sample_rate)
            buffer = None
            asr_result = model.generate(input=folder + "buffer.wav", 
                        batch_size_s=6000, 
                        hotword='增容'
                        )
            with open(folder + asr_file,'a') as f:
                f.write(str(last_id) + asr_result[0]['text'] + '\n')       
        else:
            buffer = np.concatenate((buffer, piece), axis = 0) if buffer is not None else piece
        last_id = now_id

            
# save
if buffer is not None:
    sf.write(folder + "buffer.wav", buffer, sample_rate)
    asr_result = model.generate(input=folder + "buffer.wav", 
                            batch_size_s=6000, 
                            hotword='增容'
                            )
    with open(folder + asr_file,'a') as f:
        f.write(str(last_id) + asr_result[0]['text'] + '\n')   

