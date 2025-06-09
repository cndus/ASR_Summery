# 音频转录（ASR）+ 说话人识别

## 6.10 update
- 修复了说话人识别的bug
- 最终输出回退到上一个版本的格式：伪实时部分不变；最终的输出会单独进行一个ASR，在```total_asr.txt```中给出说话人id和ASR结果。其中id和SpeakerManager.get_all_features()函数返回的list中位置对应的就是预先录入的说话人；反之，如果id大于预录入说话人列表的长度，就是新说话人。

## 4.23update：SpeakerManager
添加了预先输入说话人音频和姓名的SpeakerManager类。在进行ASR时，读取存取的说话人特征；后续可以通过调用SpeakerManager中的函数来获取id相应的说话人姓名。

详见```spks.py```

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
- ASR及说话人识别输出路径写死在“save”环节的代码中，现在是```buffer.txt```。每行代表一个短句，前面的数字是说话人ID。不同30s的音频片段的ASR不断被添加到输出文档中，之间使用空行隔开。


## 效果
在官方示例上，流式比非流式略差；在实际音频上，流式错别字较多，非流式效果一般。
