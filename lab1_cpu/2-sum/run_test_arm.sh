#!/bin/bash
sizes=(2048 4096 8192 16384 32768 65536 131072)
repeats=20  # 设定重复次数

for n in "${sizes[@]}"; do
    echo "====== Testing n=$n ======"

    # 生成数据
    ./gen_data $n input_$n.txt

    # qemu 模拟
    echo "-- Naive --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ ./naive_arm $n input_$n.txt
    done

    echo "-- Optimized01 --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ ./optimized01_arm $n input_$n.txt
    done

    echo "-- Optimized02 --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ ./optimized02_arm $n input_$n.txt
    done
    
    echo "-- Optimized03 --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ ./optimized03_arm $n input_$n.txt
    done
    rm input_$n.txt
done


