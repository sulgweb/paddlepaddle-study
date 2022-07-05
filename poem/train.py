'''
Description: 
Author: xianpengfei
LastEditors: xianpengfei
Date: 2022-07-05 17:20:34
LastEditTime: 2022-07-05 19:44:18
'''
import paddle
from paddle.static import InputSpec
from paddlenlp.metrics import Perplexity
from paddle.optimizer import AdamW
from paddle.io import DataLoader
from PoemModel import PoetryBertModel, PoetryBertModelLossCriterion
from PoemData import PoemData
import paddlenlp
import re
from paddlenlp.transformers import BertTokenizer


def data_preprocess(dataset):
    for i, data in enumerate(dataset):
        dataset.data[i] = ''.join(list(dataset[i].values()))
        dataset.data[i] = re.sub('\x02', '', dataset[i])
    return dataset

# 获取数据模型
test_dataset, dev_dataset, train_dataset = paddlenlp.datasets.load_dataset('poetry', splits=('test','dev','train'), lazy=False)
print('test_dataset 的样本数量：%d'%len(test_dataset))
print('dev_dataset 的样本数量：%d'%len(dev_dataset))
print('train_dataset 的样本数量：%d'%len(train_dataset))

# 处理数据
test_dataset = data_preprocess(test_dataset)
dev_dataset = data_preprocess(dev_dataset)
train_dataset = data_preprocess(train_dataset)
print('处理后的单样本示例：%s'%test_dataset[0])

# 对诗歌进行分词和编码
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 训练
net = PoetryBertModel('bert-base-chinese', 128)
token_ids = InputSpec((-1, 128), 'int64', 'token')
token_type_ids = InputSpec((-1, 128), 'int64', 'token_type')
input_mask = InputSpec((-1, 128), 'float32', 'input_mask')
label = InputSpec((-1, 128), 'int64', 'label')

inputs = [token_ids, token_type_ids, input_mask]
labels = [label, input_mask]

model = paddle.Model(net, inputs, labels)
model.prepare(optimizer=AdamW(learning_rate=0.0001, parameters=model.parameters()), loss=PoetryBertModelLossCriterion(), metrics=[Perplexity()])

model.summary(inputs, [input.dtype for input in inputs])
train_loader = DataLoader(PoemData(train_dataset, bert_tokenizer, 128), batch_size=128, shuffle=True)
dev_loader = DataLoader(PoemData(dev_dataset, bert_tokenizer, 128), batch_size=32, shuffle=True)
model.fit(train_data=train_loader, epochs=10, save_dir='./checkpoint', save_freq=1, verbose=1, eval_data=dev_loader, eval_freq=1)