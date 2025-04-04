#!/bin/bash
sizes=(8192 16384 32768 65536 131072 262144 524288)
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

    echo "-- Optimized01 --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        ./optimized01_x86 $n input_$n.txt
    done

    echo "-- Optimized02 --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        ./optimized02_x86 $n input_$n.txt
    done
    
    echo "-- Optimized03 --"
    for i in $(seq 1 $repeats); do
        echo "Run #$i"
        ./optimized03_x86 $n input_$n.txt
    done
    rm input_$n.txt
done


