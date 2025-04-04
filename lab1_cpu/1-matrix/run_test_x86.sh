#!/bin/bash
sizes=(80 160 320 640 1280 2560 5120)
repeats=20  # 设定重复次数

for n in "${sizes[@]}"; do
    echo "====== Testing n=$n ======"

    # 生成数据
    ./gen_data $n input_$n.txt

    echo "-- Naive --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        ./naive_x86 $n input_$n.txt
    done

    echo "-- Optimized --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        ./optimized_x86 $n input_$n.txt
    done

    rm input_$n.txt
done

