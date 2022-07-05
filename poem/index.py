'''
Description: 
Author: xianpengfei
LastEditors: xianpengfei
Date: 2022-07-05 16:18:16
LastEditTime: 2022-07-05 16:23:37
'''
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
for poem in test_dataset[0:2]:
    token_poem, _ = bert_tokenizer.encode(poem).values()
    print(poem)
    print(token_poem)
    print(''.join(bert_tokenizer.convert_ids_to_tokens(token_poem)))