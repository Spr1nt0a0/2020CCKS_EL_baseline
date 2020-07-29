# 2020CCKS_EL_baseline

## 环境
Python3 + Paddle Fluid 1.5

## 下载数据
请从[竞赛网站](https://biendata.com/competition/ccks_2020_el/data/)上下载数据，解压后放在./data/basic_data/目录

解压后,目录./data/basic_data/ 中包含文件: 
    
    dev.json
    kb.json
    test.json
    train.json
    eval.py
    README
    CCKS 2020 Entity Linking License.docx
    
## 下载预训练的Ernie模型
下载ERNIE1.0 Base模型，并将其解压到./pretrained_model/

cd ./pretrained_mdoel/

wget --no-check-certificate https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz

tar -zxvf ERNIE_1.0_max-len-512.tar.gz

解压后,路径./pretrained_model/ERNIE_1.0_max-len-512 中包含文件：
    
    ernie_config.json
    params
    vocab.txt

## 数据格式转换
生成的数据在./data/generated/目录中

cd ./data/

python data_process.py
## 训练
sh ./script/train.sh

默认情况在，模型将保存到./checkpoints/

训练过程中会打印准确率和f1

训练和预测前注意调整python路径和数据集路径

建议使用10000条以下的数据作为验证集，以免验证集过大导致整体耗时增加
## 预测
调整预测脚本中的模型路径，运行：

sh ./script/predict.sh

预测结果将以与原始数据集的相同的格式写入json文件（与最终官方评估格式相同）。预测结果路径为./data/generated/test_pred.json
