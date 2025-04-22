
from funasr import AutoModel
import torch
import soundfile as sf
import torchaudio as ta
from math import ceil
import math
import numpy as np
from spks import SpeakerManager


folder = 'eg2/'
asr_file = 'buffer.txt'
cache_path = "/home/xhyin/.cache/modelscope/hub/iic/"
#------------ Load data------------
waveform, sample_rate = sf.read(folder + "eg16k.wav")
if len(waveform.shape) == 1:
    waveform = np.expand_dims(waveform, axis=1)
elif waveform.shape[1] > 1:
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

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}


supports = {
    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
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
pretrained_model = save_dir / conf['model_pt']
if not pathlib.Path(pretrained_model).exists():
    cache_dir = snapshot_download(
                args.model_id,
                revision=conf['revision'],
                )
    cache_dir = pathlib.Path(cache_dir)

    embedding_dir = save_dir / 'embeddings'
    embedding_dir.mkdir(exist_ok=True, parents=True)

# link
    print(f'[INFO]: {pretrained_model} not found. Linking files...')
    download_files = ['examples', conf['model_pt']]
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

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
if pathlib.Path(cache_path).exists():
    model = AutoModel(model=cache_path + "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  
                  vad_model=cache_path + "speech_fsmn_vad_zh-cn-16k-common-pytorch", 
                  spk_model="cam++",#speech_campplus_sv_zh-cn_16k-common
                  punc_model=cache_path + "punc_ct-transformer_cn-en-common-vocab471067-large", disable_update=True,
                  )
else:
    model = AutoModel(model="paraformer-zh",  
                  vad_model="fsmn-vad", 
                  spk_model="cam++",
                  punc_model="ct-punc", disable_update=True,
                  )

# -------------------computing...----------------
low_threshold = 0.6
high_threshold = 0.84
max_buffer_length = 10 * sample_rate
FRAME_PER_SECOND = 1000

buffer = None
buffer_piece_id = [-1, -1] # record correspoding piece id in buffer: (start id, end id)

spks = [] # shape=[spk num, emb length]
spks_num = []
last_id = -1
now_id = -1
stage_id = -1
stage_piece = None

spkmanager = SpeakerManager()
spks = spkmanager.get_all_features()
spks_num = [i+1 for i in range(len(spks))]

for i in range(piece_num):
    if i != piece_num - 1:
        sf.write(folder + "piece.wav", waveform[i*stride: (i+1)*stride],sample_rate)
    else:
        sf.write(folder + "piece.wav", waveform[i*stride: ],sample_rate)
    piece, sample_rate = sf.read(folder + "piece.wav")
    emb = compute_embedding(torch.from_numpy(piece).unsqueeze(0))

    # 伪实时
    asr_result = model.generate(input=folder + "piece.wav", 
                        batch_size_s=6000, 
                        hotword='增容'
                        )
    try:
        with open(folder + '伪实时.txt','a') as f:
            f.write('piece id = ' + str(i) + '##' + asr_result[0]['text'] + '\n')
    except:
        print('warning: asr result=',asr_result)
        continue

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
        buffer = piece
        buffer_piece_id = [i, i]
    else:
    # not first piece
        ## set spk id
        if min(score) < high_threshold:
            now_id = id = np.argmin(np.array(score))
            spks[id] = (spks[id]*spks_num[id] + emb) / (spks_num[id] + 1)
            spks_num[id] += 1
        else:
            spks.append(emb)
            spks_num.append(1)
            now_id = len(spks) - 1

        ## set buffer
        if stage_id != -1:
        # spk may change in last piece
            if last_id == now_id:
            # spk change: last&now is a spk, before is another spk
                sf.write(folder + "buffer.wav", buffer, sample_rate)
                asr_result = model.generate(input=folder + "buffer.wav", 
                            batch_size_s=6000, 
                            hotword='增容'
                            )
                with open(folder + asr_file,'a') as f:
                    f.write('piece id = ' + str(buffer_piece_id) + ', spk id =' + str(stage_id) + asr_result[0]['text'] + '\n')   
                buffer = np.concatenate((stage_piece, piece), axis = 0)    
                buffer_piece_id = [i-1, i]
            else:
                if now_id == stage_id:
                # spk not change
                    buffer = np.concatenate((buffer, stage_piece), axis = 0)
                    buffer = np.concatenate((buffer, piece), axis = 0)
                    buffer_piece_id[1] = i
                else:
                # spk change: last, now, before are different spk
                    sf.write(folder + "buffer.wav", buffer, sample_rate)
                    asr_result = model.generate(input=folder + "buffer.wav", 
                                batch_size_s=6000, 
                                hotword='增容'
                                )
                    with open(folder + asr_file,'a') as f:
                        f.write('piece id = ' + str(buffer_piece_id) + ', spk id =' + str(stage_id) + asr_result[0]['text'] + '\n')   

                    sf.write(folder + "buffer.wav", stage_piece, sample_rate)
                    asr_result = model.generate(input=folder + "buffer.wav", 
                                batch_size_s=6000, 
                                hotword='增容'
                                )
                    with open(folder + asr_file,'a') as f:
                        f.write('piece id = ' + str([i-1,i-1]) + ', spk id =' + str(last_id) + asr_result[0]['text'] + '\n')   
                    
                    buffer = piece
                    buffer_piece_id = [i, i]

            stage_id = -1
            stage_piece = None
        elif buffer is not None and last_id != now_id:
        # spk may change in this piece
            if min(score) < low_threshold:
                # confidant that spkm change
                sf.write(folder + "buffer.wav", buffer, sample_rate)
                asr_result = model.generate(input=folder + "buffer.wav", 
                            batch_size_s=6000, 
                            hotword='增容'
                            )
                with open(folder + asr_file,'a') as f:
                    f.write('piece id = ' + str(buffer_piece_id) + ', spk id =' + str(last_id) + asr_result[0]['text'] + '\n')  

                buffer = piece   
                buffer_piece_id = [i, i] 
            else:
                # not confidant: wait for next piece to check
                stage_id = last_id
                stage_piece = piece
        elif buffer is not None and len(buffer) > max_buffer_length:
        # update ASR bacause buffer is too long
            buffer = np.concatenate((buffer, piece), axis = 0)
            buffer_piece_id[1] = i
            sf.write(folder + "buffer.wav", buffer, sample_rate)
            asr_result = model.generate(input=folder + "buffer.wav", 
                        batch_size_s=6000, 
                        hotword='增容'
                        )
            with open(folder + asr_file,'a') as f:
                f.write('piece id = ' + str(buffer_piece_id) + ', spk id =' + str(last_id) + asr_result[0]['text'] + '\n')       

            buffer = None  
            buffer_piece_id = [i+1, -1]  
        else:
        # nothing special, simply add piece to buffer
            buffer = np.concatenate((buffer, piece), axis = 0) if buffer is not None else piece
            buffer_piece_id[1] = i
        last_id = now_id

            
# save for final piece
if buffer is not None:
    sf.write(folder + "buffer.wav", buffer, sample_rate)
    asr_result = model.generate(input=folder + "buffer.wav", 
                            batch_size_s=6000, 
                            hotword='增容'
                            )
    with open(folder + asr_file,'a') as f:
        f.write('piece id = ' + str(buffer_piece_id) + ', spk id =' + str(last_id) + asr_result[0]['text'] + '\n')   

# # total ASR
# asr_result = model.generate(input=folder + "eg16k.wav", 
#                         batch_size_s=6000, 
#                         hotword='增容'
#                         )
# last_id = -1
# with open(folder + "total_asr.txt",'w') as f:
#     for info in asr_result[0]['sentence_info']:            
#         if last_id != info['spk']:
#             last_id = info['spk']
#             f.write('\n' + str(info['spk']) + ' ## ')
#         f.write(info['text'])   

