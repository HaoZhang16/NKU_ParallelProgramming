#include <queue>
#include <arm_neon.h>
#include "pq_simd_scan.h"

// 计算查询向量各段与中心表的点积，量化并存储成4个uint8x16_t表
void fs_pre_calculate_quantized(float* center, float* query, uint8x16_t tables[4], size_t center_num, size_t center_vecdim) {
    float tmp[16 * 4]; // 16类 × 4段
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < center_num; ++j) {
            // center的布局是段内连续的
            tmp[j + i * center_num] = InnerProductSIMDNeon(center + (j + i * center_num) * center_vecdim, query + i * center_vecdim, center_vecdim);
        }
    }

    // 动态估计qmin, qmax
    float qmin = tmp[0], qmax = tmp[0];
    for (int i = 0; i < 16 * 4; ++i) {
        qmin = std::min(qmin, tmp[i]);
        qmax = std::max(qmax, tmp[i]);
    }
    // 为了保险稍微扩一点点
    float margin = (qmax - qmin) * 0.05f;
    qmin -= margin;
    qmax += margin;


    uint8_t quantized[16 * 4];
    QuantizeSIMD(tmp, quantized, 16 * 4, qmin, qmax);

    // 将每段的数据分别打包
    for (int i = 0; i < 4; ++i) {
        tables[i] = vld1q_u8(quantized + i * 16);
    }
}

// 主查询函数
std::priority_queue<std::pair<float, uint32_t>> fs_simd_search(uint8_t* base, float* center, float* query,
    size_t base_number, size_t vecdim, size_t k, size_t center_num, size_t center_vecdim, float* base_full) {

    std::priority_queue<std::pair<float, uint32_t>> q;

    // 预处理
    uint8x16_t tables[4];
    fs_pre_calculate_quantized(center, query, tables, center_num, center_vecdim);

    size_t rerank = k * 500;
    std::priority_queue<std::pair<uint16_t, uint32_t>> candidates;

    for (size_t i = 0; i < base_number; i += 16) { // 每批处理16条向量
        uint8_t idx0_raw[16], idx1_raw[16], idx2_raw[16], idx3_raw[16];
    
        for (int j = 0; j < 16 && (i + j) < base_number; ++j) {
            uint8_t* ptr = base + (i + j) * 4;  // 每个向量4个字节
    
            idx0_raw[j] = ptr[0] & 0x0F;  // 第0段
            idx1_raw[j] = ptr[1] & 0x0F;  // 第1段
            idx2_raw[j] = ptr[2] & 0x0F;  // 第2段
            idx3_raw[j] = ptr[3] & 0x0F;  // 第3段
        }
    
        uint8x16_t idx0 = vld1q_u8(idx0_raw);
        uint8x16_t idx1 = vld1q_u8(idx1_raw);
        uint8x16_t idx2 = vld1q_u8(idx2_raw);
        uint8x16_t idx3 = vld1q_u8(idx3_raw);
    
        // 使用vqtbl1q_u8快速查表
        uint8x16_t val0 = vqtbl1q_u8(tables[0], idx0);
        uint8x16_t val1 = vqtbl1q_u8(tables[1], idx1);
        uint8x16_t val2 = vqtbl1q_u8(tables[2], idx2);
        uint8x16_t val3 = vqtbl1q_u8(tables[3], idx3);
    
        // 先将每两个段累加成u16
        uint16x8_t partial0 = vaddl_u8(vget_low_u8(val0), vget_low_u8(val1));
        uint16x8_t partial1 = vaddl_u8(vget_low_u8(val2), vget_low_u8(val3));

        // 再把partial结果相加
        uint16x8_t sum_low = vaddq_u16(partial0, partial1);

        uint16x8_t partial2 = vaddl_u8(vget_high_u8(val0), vget_high_u8(val1));
        uint16x8_t partial3 = vaddl_u8(vget_high_u8(val2), vget_high_u8(val3));

        uint16x8_t sum_high = vaddq_u16(partial2, partial3);
    
        uint16_t result[16];
        vst1q_u16(result, sum_low);
        vst1q_u16(result + 8, sum_high);
    
        for (int j = 0; j < 16 && (i + j) < base_number; ++j) {
            uint16_t raw_dis = result[j];
            uint16_t dis = 65535 - raw_dis; 
        
            if (candidates.size() < rerank) {
                candidates.push({dis, i + j});
            } else if (dis < candidates.top().first) {
                candidates.push({dis, i + j});
                candidates.pop();
            }
        }
    }

    // rerank阶段，全精度重新计算
    while (!candidates.empty()) {
        auto cand = candidates.top();
        candidates.pop();
        uint32_t idx = cand.second;
        float dis = InnerProductSIMDNeon(base_full + idx * vecdim, query, vecdim);
        dis = 1.0f - dis;

        if (q.size() < k) {
            q.push({dis, idx});
        } else if (dis < q.top().first) {
            q.push({dis, idx});
            q.pop();
        }
    }

    return q;
}