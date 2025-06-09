from funasr import AutoModel
import torch
import soundfile as sf
import torchaudio as ta
from math import ceil
import json
import numpy as np


#------------ Load data------------


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



class SpeakerManager:
    def __init__(self, storage_file='speakers.json'):
        self.storage_file = storage_file
        self.speakers = self._load_speakers()

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
        self.embedding_model = dynamic_import(model['obj'])(**model['args'])
        self.embedding_model.load_state_dict(pretrained_state)
        self.embedding_model.eval()

        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def compute_embedding(self, wav):
        # compute feat
        feat = self.feature_extractor(wav).unsqueeze(0)#.to(device)
        # compute embedding
        with torch.no_grad():
            embedding = self.embedding_model(feat.float()).detach().squeeze(0).cpu().numpy()
        return embedding


    def _load_speakers(self):
        # 从文件加载说话人数据
        if pathlib.Path(self.storage_file).exists():
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_speakers(self):
        # 将说话人数据保存到文件
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.speakers, f, ensure_ascii=False, indent=4)

    def add_speaker(self, wav, name):
        # 添加说话人
        embedding = self.compute_embedding(wav)
        self.speakers[name] = embedding.tolist()  # 将numpy数组转换为列表以便存储
        self._save_speakers()

    def delete_speaker(self, name):
        # 删除说话人
        if name in self.speakers:
            del self.speakers[name]
            self._save_speakers()
        else:
            print(f"[ERROR]: Speaker '{name}' not found.")

    def get_all_features(self):
        # 返回所有说话人特征
        spk_feature_list = [np.array(embedding) for embedding in self.speakers.values()]
        return spk_feature_list

    def query_speaker(self, index):
        # 根据索引查询说话人姓名
        if 0 <= index < len(self.speakers):
            return list(self.speakers.keys())[index]
        else:
            print(f"[ERROR]: Index {index} out of range.")
            return None
        
if __name__ == "__main__":
    manager = SpeakerManager()

    # 添加说话人
    wav, sample_rate = sf.read("/home/xhyin/ASR_Summery/eg5/cut_20250609_193910.wav")
    wav = torch.from_numpy(wav)[:,0]
    print(wav.shape)
    manager.add_speaker(wav, name="主持人")

    # 所有特征向量
    features = manager.get_all_features()

    # 查询说话人姓名
    name_at_0 = manager.query_speaker(0)

    # # 删除一个说话人
    # manager.delete_speaker("张三")
