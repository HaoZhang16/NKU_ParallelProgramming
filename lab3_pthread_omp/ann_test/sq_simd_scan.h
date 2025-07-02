#include <queue>
#include <arm_neon.h>
#include "plain_simd_scan.h"

void Quantize(const float* input, uint8_t* output, size_t dim, float min_val, float max_val) {
    float scale = 255.0f / (max_val - min_val);

    for (size_t i = 0; i < dim; ++i) {
        float normalized = (input[i] - min_val) * scale;

        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 255.0f) normalized = 255.0f;

        output[i] = static_cast<uint8_t>(normalized);
    }
}

void QuantizeSIMD(const float* input, uint8_t* output, size_t dim, float min_val, float max_val) {
    // 先构造8位参数
    float scale = 255.0f / (max_val - min_val);
    
    simd8float32 scale_vec(scale);
    simd8float32 min_val_vec(min_val);

    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    float32x4_t max_vec = vdupq_n_f32(255.0f);

    for (size_t i = 0; i < dim; i += 8) {
        // 载入8个float
        simd8float32 x(input + i);

        // normalized = (input[i] - min_val) * scale
        simd8float32 normalized;
        normalized = (x - min_val_vec) * scale_vec;

        // 限制到[0, 255]
        normalized.data.val[0] = vmaxq_f32(normalized.data.val[0], zero_vec);
        normalized.data.val[0] = vminq_f32(normalized.data.val[0], max_vec);
        normalized.data.val[1] = vmaxq_f32(normalized.data.val[1], zero_vec);
        normalized.data.val[1] = vminq_f32(normalized.data.val[1], max_vec);

        // 保存到临时数组
        float tmp[8];
        normalized.storeu(tmp);

        // 转成uint8_t存回output
        for (int j = 0; j < 8; ++j) {
            output[i + j] = static_cast<uint8_t>(tmp[j]);
        }
    }
}


float InnerProductSIMDNeonQuantized(const uint8_t* aq, const uint8_t* bq, size_t vecdim, float scale, float offset) {
    assert(vecdim % 16 == 0); // 确保是16的倍数

    uint32x4_t total_dot = vdupq_n_u32(0);
    uint32x4_t total_a = vdupq_n_u32(0);
    uint32x4_t total_b = vdupq_n_u32(0);

    for (size_t i = 0; i < vecdim; i += 16) {
        uint8x16_t va = vld1q_u8(aq + i);
        uint8x16_t vb = vld1q_u8(bq + i);

        // 低8位
        uint8x8_t va_low = vget_low_u8(va);
        uint8x8_t vb_low = vget_low_u8(vb);

        // 高8位
        uint8x8_t va_high = vget_high_u8(va);
        uint8x8_t vb_high = vget_high_u8(vb);

        // 使用vmlal_u8直接乘加
        uint16x8_t mul_low = vmull_u8(va_low, vb_low);
        uint16x8_t mul_high = vmull_u8(va_high, vb_high);

        // 再累加到uint32x4
        total_dot = vpadalq_u16(total_dot, mul_low);
        total_dot = vpadalq_u16(total_dot, mul_high);

        // 计算Σaq
        uint16x8_t va_low_u16 = vmovl_u8(va_low);
        uint16x8_t va_high_u16 = vmovl_u8(va_high);
        total_a = vpadalq_u16(total_a, va_low_u16);
        total_a = vpadalq_u16(total_a, va_high_u16);

        // 计算Σbq
        uint16x8_t vb_low_u16 = vmovl_u8(vb_low);
        uint16x8_t vb_high_u16 = vmovl_u8(vb_high);
        total_b = vpadalq_u16(total_b, vb_low_u16);
        total_b = vpadalq_u16(total_b, vb_high_u16);
    }

    // 横向求和
    uint32_t dot = vaddvq_u32(total_dot);
    uint32_t sum_a = vaddvq_u32(total_a);
    uint32_t sum_b = vaddvq_u32(total_b);

    // 反量化点积
    float inv_scale_sq = 1.0f / (scale * scale);
    float offset_scale = offset / scale;
    float term1 = dot * inv_scale_sq;
    float term2 = offset_scale * (sum_a + sum_b);
    float term3 = static_cast<float>(vecdim) * offset * offset;

    return term1 - term2 + term3;
}

std::priority_queue<std::pair<float, uint32_t>> sq_simd_search(uint8_t* base, float* query, size_t base_number, size_t vecdim, size_t k, float* base_full) {
    std::priority_queue<std::pair<float, uint32_t> > q;

	float min_val = -1.0f;
	float max_val = 1.0f;
    float scale = 255.0f / (max_val - min_val);
    float offset = -min_val;
    uint8_t* quantized_query = new uint8_t[vecdim];

	// 先量化查询向量
	QuantizeSIMD(query, quantized_query, vecdim, min_val, max_val);

    // 存储所有找到的候选近邻
    size_t rerank = (size_t)(k * 2); // 设置rerank
    std::priority_queue<std::pair<float, uint32_t>> candidates;

	for(int i = 0; i < base_number; ++i){
		float dis = InnerProductSIMDNeonQuantized(base + i * vecdim, quantized_query, vecdim, scale, offset);
        
        dis = 1-dis;

        // 存储候选项
        if(candidates.size() < rerank){
         candidates.push({dis, i});
        }else{
            if(dis < candidates.top().first){
                candidates.push({dis, i});
                candidates.pop();
            }
        }
    }

    // 进行全精度重排序
    while(!candidates.empty()){
        auto cand = candidates.top();
        candidates.pop();
        int i = cand.second;
        float dis = InnerProductSIMDNeon(base_full + i * vecdim, query, vecdim);
        dis = 1 - dis;

        // 维护最大堆
        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    delete[] quantized_query;
    return q;
}
