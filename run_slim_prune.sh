#!/bin/bash

# 运行程序使用的GPU
device=1

# 每层最少要保留的通道比例
layer_keep=0.01

epochs=300
batch_size=16

# 数据集
data=data/voc.data

# 模型配置文件
pre_model=yolov3
model=yolov3-voc

# 稀疏率
s=0.001

# 保存运行结果的目录
prefix=exp_voc_slim


echo "device=${device}"
echo "layer_keep=${layer_keep}"
echo "epochs=${epochs}"
echo "batch_size=${batch_size}"
echo "data=${data}"
echo "model=${model}"
echo "s=${s}"

mkdir ${prefix}

echo "################ start baseline"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    --cfg cfg/${model}.cfg \
    --weights weights/${pre_model}.weights \
    --data ${data} \
    --batch-size ${batch_size} \
    --epochs ${epochs}
echo "################ end baseline"

# 将 weights 目录下 基础训练 产生的 results,tb,weights 保存
# results
mkdir -p ${prefix}/baseline/results
mv -f results.txt ${prefix}/baseline/results
mv -f results.png ${prefix}/baseline/results
mv -f results.json ${prefix}/baseline/results

# tb
mkdir -p ${prefix}/baseline/runs
mv -f runs/* ${prefix}/baseline/runs

# weights
mkdir -p ${prefix}/baseline/weights
rm -rf weights/backup*
cp -f weights/last.pt ${prefix}/baseline/weights
mv -f weights/best.pt ${prefix}/baseline/weights



# convert weights/last.pt to weights/last.weights
python -c "from models import *; convert('cfg/${model}.cfg', 'weights/last.pt')"
mv -f weights/last.weights ${prefix}/baseline/weights
# 删除 weights 目录下 稀疏训练 产生的 last.pt
rm -rf weights/last.pt


echo "################ start sp"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    --cfg cfg/${model}.cfg \
    --weights ${prefix}/baseline/weights/last.weights \
    --data ${data} \
    --batch-size ${batch_size} \
    --epochs ${epochs} \
    -sr \
    --s ${s} \
    --prune 1
echo "################ end sp"

# 将 稀疏训练 产生的 results,tb,weights 保存
# results
mkdir -p ${prefix}/sp/results
mv -f results.txt ${prefix}/sp/results
mv -f results.png ${prefix}/sp/results
mv -f results.json ${prefix}/sp/results

# tb
mkdir -p ${prefix}/sp/runs
mv -f runs/* ${prefix}/sp/runs

# weights
mkdir -p ${prefix}/sp/weights
rm -rf weights/backup*
mv -f weights/last.pt ${prefix}/sp/weights
mv -f weights/best.pt ${prefix}/sp/weights


for global_percent in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    # 将稀疏训练得到的权重 last.pt 复制到 weights 目录下
    cp -f ${prefix}/sp/weights/last.pt weights/

    echo "################ start slim-prune.py"
    CUDA_VISIBLE_DEVICES=${device} python slim_prune.py \
        --cfg cfg/${model}.cfg \
        --data ${data} \
        --weights weights/last.pt \
        --global_percent ${global_percent} \
        --layer_keep ${layer_keep}
    echo "################ end slim-prune.py"

    # 将 剪枝 产生的 cfg,weights 保存
    # cfg
    mkdir -p ${prefix}/percent-${global_percent}
    mv -f cfg/prune_${global_percent}_keep_${layer_keep}_${model}.cfg ${prefix}/percent-${global_percent}/

    # weights
    mv -f weights/prune_${global_percent}_keep_${layer_keep}_last.weights ${prefix}/percent-${global_percent}/



    echo "################ start finetune"
    CUDA_VISIBLE_DEVICES=${device} python train.py \
        --cfg ${prefix}/percent-${global_percent}/prune_${global_percent}_keep_${layer_keep}_${model}.cfg  \
        --weights ${prefix}/percent-${global_percent}/prune_${global_percent}_keep_${layer_keep}_last.weights \
        --data ${data} \
        --batch-size ${batch_size} \
        --epochs ${epochs}
    echo "################ end finetune"

    # 将 微调 产生的 results,tb,weights 保存
    # results
    mkdir -p ${prefix}/percent-${global_percent}/results
    mv -f results.txt ${prefix}/percent-${global_percent}/results
    mv -f results.png ${prefix}/percent-${global_percent}/results
    mv -f results.json ${prefix}/percent-${global_percent}/results

    # tb
    mkdir -p ${prefix}/percent-${global_percent}/runs
    mv -f runs/* ${prefix}/percent-${global_percent}/runs

    # weights
    mkdir -p ${prefix}/percent-${global_percent}/weights
    rm -rf weights/backup*
    mv -f weights/last.pt ${prefix}/percent-${global_percent}/weights
    mv -f weights/best.pt ${prefix}/percent-${global_percent}/weights
done

rm -rf runs

echo "################ finish baseline, sp, prune, finetune!!!!!!!!!!!!"
