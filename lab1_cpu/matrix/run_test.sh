#!/bin/bash
sizes=(40)
for n in "${sizes[@]}"; do
    echo "====== Testing n=$n ======"
    
    ./gen-data $n input_$n.txt

    # 预热缓存
#    /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ naive_arm $n input_$n.txt > /dev/null
#    /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ optimized_arm $n input_$n.txt > /dev/null

    # 正式测试
    echo "-- Naive --"
    time /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ naive_arm $n input_$n.txt > /dev/null
    echo "-- Optimized --"
    time /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ optimized_arm $n input_$n.txt > /dev/null

    rm input_$n.txt
done
