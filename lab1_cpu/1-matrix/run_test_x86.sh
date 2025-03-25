#!/bin/bash
sizes=(500 1000 2000 4000 8000 16000)
for n in "${sizes[@]}"; do
    echo "====== Testing n=$n ======"
    
    ./gen_data $n input_$n.txt
    
    # 预热缓存
#./naive $n input_$n.txt > /dev/null
#./optimized $n input_$n.txt > /dev/null
    
    # 正式测试
    echo "-- Naive --"
    time ./naive $n input_$n.txt > /dev/null
    echo "-- Optimized --"
    time ./optimized $n input_$n.txt > /dev/null
    
    rm input_$n.txt
done
