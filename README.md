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

**特别注意：类别标签是从１开始的，因为在后面训练的时候需要做pad 0 的操作,为了避免混淆。**

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

# 使用的模型 可选cnn, bilstm, clstm
MODEL=cnn

# 中间文件输出路径
DATA_OUT_DIR=./datasets/dbpedia/


# 模型输出路径
MODEL_OUT_DIR=./results/dbpedia/
```

### 2.embedding
生成word2vec的训练数据
```bash
./main.sh pre
```

训练词向量
```bash
./main.sh vec
```

### 3.map file

这一步产生需要的映射文件

```bash
./main.sh map
```

### 4.tfrecord

产生tfrecord 文件

```bash
./main.sh data
```

### 5.train
模型训练
```bash
./main.sh train
```

### 6.模型导出
导出成pb文件，可用Java，Go语言读取

```bash
./main export
```

### 模型使用
在```predict.py```中有例子，读取上面训练好导出的模型，和产生的```vocab.json```文件

TextRNN、TextCNN,CLstm 模型能共用这个模块


## todo
* 模型导出, 模型条用代码
* 根据最新的tensorflow重构代码
* 修改tfrecord 文件的格式，产生多分而不是一份
* 添加tensorboard　

待实现文本分类模型实现
 1. [HAM](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
 2. [RCNN](https://scholar.google.com.hk/scholar?q=Recurrent+Convolutional+Neural+Networks+for+Text+Classification&hl=zh-CN&as_sdt=0&as_vis=1&oi=scholart&sa=X&ved=0ahUKEwjpx82cvqTUAhWHspQKHUbDBDYQgQMIITAA)
 3. Recurrent Entity Network
 3. Dynamic Memory Network
