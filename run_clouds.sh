#!/bin/bash
echo "=== 开始生成 Train 数据集 ==="
python generate_synthetic_clouds.py --input-dir /mnt/mydisk/zwj/zzy/EMRDM-main/datasets/archive/TrainData/train/images_cropped --output-dir /mnt/mydisk/zwj/zzy/EMRDM-main/datasets/Cloudy_iSAID/train --copy-label --preserve-relative-paths

echo "=== 开始生成 Val 数据集 ==="
python generate_synthetic_clouds.py --input-dir /mnt/mydisk/zwj/zzy/EMRDM-main/datasets/archive/ValidationData/val/images_cropped --output-dir /mnt/mydisk/zwj/zzy/EMRDM-main/datasets/Cloudy_iSAID/val --copy-label --preserve-relative-paths

echo "=== 开始生成 Test 数据集 ==="
python generate_synthetic_clouds.py --input-dir /mnt/mydisk/zwj/zzy/EMRDM-main/datasets/archive/TestData/TestData/images_cropped --output-dir /mnt/mydisk/zwj/zzy/EMRDM-main/datasets/Cloudy_iSAID/test --copy-label --preserve-relative-paths

echo "=== 全部生成完毕！ 🎉 ==="
