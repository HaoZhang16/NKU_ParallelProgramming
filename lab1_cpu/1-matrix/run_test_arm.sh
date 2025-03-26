#!/bin/bash
sizes=(20 640)
repeats=20  # 设定重复次数

for n in "${sizes[@]}"; do
    echo "====== Testing n=$n ======"

    # 生成数据
    ./gen_data $n input_$n.txt

    # 正式测试（添加 qemu 前缀）
    echo "-- Naive --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ naive_arm $n input_$n.txt
    done

    echo "-- Optimized --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ optimized_arm $n input_$n.txt
    done

    rm input_$n.txt
done

