# 音频转录（ASR）+ 说话人识别
## 使用的开源模型
- 使用来自[FunASR](https://github.com/modelscope/FunASR/blob/main/README_zh.md)的开源模型，进行中文语音识别。
- 使用[3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker)的开源模型进行说话人识别。

## run
运行```python ASR_nonStream.py```，使用预训练模型进行**非实时**语音识别。

输入要求：16kHz，双声道，```.wav```后缀的文件

路径设置：
- 数据文件夹通过参数```folder```设置
- 会议整体音频路径写死在“Load data”环节的代码里，现在是```eg16k.wav```。（实际使用场景中，这个音频并不存在。）
- 每一小段的音频存放在```piece.wav```中，模拟来自客户端的输入，程序读取该文件的音频进行处理。
- ASR及说话人识别输出路径写死在“save”环节的代码中，现在是```eg_funasr_nonStream_spk.txt```。每行代表一个短句，前面的数字是说话人ID。不同30s的音频片段的ASR不断被添加到输出文档中，之间使用空行隔开。


## 效果
在官方示例上，流式比非流式略差；在实际音频上，流式错别字较多，非流式效果一般。
