# CoolNLTK

文本分类工具集

## 已实现模型
1. [TextCNN](https://arxiv.org/abs/1408.5882)
2. [TextRNN](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
3. [CLstm](https://arxiv.org/abs/1602.06291)

## 模型训练

### 1.train file
使用和[fastText](https://github.com/facebookresearch/fastText)一样的数据输入

**测试数据可以从fastText的代码中下载然后copy到```./datasets/dbpedia```目录下**

一个例子如下:
```
__label__7 , joseph purdy homestead
__label__13 , forever young ( 1992 film )
__label__11 , nepenthes ' boca rose
__label__6 , mv eilean bhearnaraigh

```
在```train/main.sh```指定相关的训练样本路径

```shell

TRAIN_FILE=./datasets/dbpedia/dbpedia.train
TEST_FILE=./datasets/dbpedia/dbpedia.test

# 使用的模型
MODEL=cnn

# 中间文件输出路径
DATA_OUT_DIR=./datasets/dbpedia/


# 模型输出路径
MODEL_OUT_DIR=./results/dbpedia/
```

### 2.embedding
生成word2vec的训练数据
```./main.sh pre```

训练词向量
```
./main.sh vec
```

### 3.map file

这一步产生需要的映射文件

```
./main.sh map
```

### 4.tfrecord

产生tfrecord 文件

```
./main.sh data
```

### 5.train
模型训练
```
./main.sh train
```


## todo
* 模型导出, 模型条用代码
* 其他文本分类模型实现
* 根据最新的tensorflow重构代码
* 修改tfrecord 文件的格式，产生多分而不是一份
* 添加tensorboard　


