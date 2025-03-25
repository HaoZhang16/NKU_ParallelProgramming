#!/bin/bash

sizes=(2048 4096 8192 16384 32768 65536)
REPEATS=20

# 生成测试数据
./gen_data "${sizes[@]}"

echo "repeats:${REPEATS}" 

# 运行测试
for n in "${sizes[@]}"; do
    input_file="data_n${n}.txt"
    echo "====== Testing n=$n ======"

    run_test() {
        local prog=$1
        echo "-- $prog --"
        total_time=0

        # 这里要利用bash的time指令获取时间 处理比较麻烦
        for ((i=1; i<=$REPEATS; i++)); do
            export LC_NUMERIC=C  # 强制使用小数点格式
            
            time_output=$( { time /usr/bin/qemu-aarch64 -L /usr/aarch64-linux-gnu/ ${prog}_arm $input_file > /dev/null; } 2>&1 )

            # 预处理 time 输出，去除异常字符，确保格式统一
            real_time=$(echo "$time_output" | grep "^real" | sed -E 's/[^0-9m.s]//g' | awk '{print $1}')

            if [[ "$real_time" =~ ([0-9]+)m([0-9.]+)s ]]; then
                min=${BASH_REMATCH[1]}
                sec=${BASH_REMATCH[2]}

                # 确保 sec 是有效的小数格式
                [[ "$sec" == .* ]] && sec="0$sec"

                # 计算总时间
                total_time=$(awk "BEGIN {print $total_time + ($min*60) + $sec}")

            else
                echo "Error parsing time output: $time_output"
                exit 1
            fi
        done
        avg_time=$(awk "BEGIN {print $total_time / $REPEATS}")
		printf "Total time: %.4fs\n" $total_time
		printf "Average time: %.4fs\n" $avg_time
    }

    run_test "naive"
    run_test "optimized01"
    run_test "optimized02"
    run_test "optimized03"

    rm $input_file
done

echo "complete"

