#include <queue>
#include <arm_neon.h>

template<typename T>
T* LoadQuantizedData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open quantized file " << data_path << "\n";
        return nullptr;
    }

    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);

    T* data = new T[n * d];
    fin.read((char*)data, sizeof(T) * n * d);
    fin.close();

    std::cerr << "load quantized data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number: " << n << "  size_per_element: " << sizeof(T) << "\n";

    return data;
}

void Quantize(const float* input, uint8_t* output, size_t dim, float min_val, float max_val) {
    float scale = 255.0f / (max_val - min_val);

    for (size_t i = 0; i < dim; ++i) {
        float normalized = (input[i] - min_val) * scale;

        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 255.0f) normalized = 255.0f;

        output[i] = static_cast<uint8_t>(normalized);
    }
}

float InnerProductSIMDNeonQuantized(const uint8_t* aq, const uint8_t* bq, size_t vecdim, float scale, float offset) {
    assert(vecdim % 16 == 0); // 确保维度是16的倍数

    uint32_t total_dot = 0;
    uint32_t total_a = 0;
    uint32_t total_b = 0;

    for (size_t i = 0; i < vecdim; i += 16) {
        uint8x16_t va = vld1q_u8(aq + i);
        uint8x16_t vb = vld1q_u8(bq + i);

        // 转换成 uint16x8_t 进行乘法计算
        uint16x8_t va_low = vmovl_u8(vget_low_u8(va));
        uint16x8_t va_high = vmovl_u8(vget_high_u8(va));
        uint16x8_t vb_low = vmovl_u8(vget_low_u8(vb));
        uint16x8_t vb_high = vmovl_u8(vget_high_u8(vb));

        // 乘法：逐位相乘
        uint32x4_t mul_ll = vmull_u16(vget_low_u16(va_low), vget_low_u16(vb_low));
        uint32x4_t mul_lh = vmull_u16(vget_high_u16(va_low), vget_high_u16(vb_low));
        uint32x4_t mul_hl = vmull_u16(vget_low_u16(va_high), vget_low_u16(vb_high));
        uint32x4_t mul_hh = vmull_u16(vget_high_u16(va_high), vget_high_u16(vb_high));

        // 累加点积结果
        total_dot += vaddvq_u32(mul_ll) + vaddvq_u32(mul_lh) + vaddvq_u32(mul_hl) + vaddvq_u32(mul_hh);

        // 求和 Σaq 和 Σbq
        total_a += vaddlvq_u8(va);
        total_b += vaddlvq_u8(vb);
    }

    // 反量化点积
    float inv_scale_sq = 1.0f / (scale * scale);
    float offset_scale = offset / scale;
    float term1 = total_dot * inv_scale_sq;
    float term2 = offset_scale * (total_a + total_b);
    float term3 = static_cast<float>(vecdim) * offset * offset;

    return term1 - term2 + term3;
}

std::priority_queue<std::pair<float, uint32_t>> sq_simd_search(uint8_t* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

	float min_val = -1.0f;
	float max_val = 1.0f;
    float scale = 255.0f / (max_val - min_val);
    float offset = -min_val;
    uint8_t* quantized_query = new uint8_t[vecdim];

	// 先量化查询向量
	Quantize(query, quantized_query, vecdim, min_val, max_val);

	for(int i = 0; i < base_number; ++i){
		float dis = InnerProductSIMDNeonQuantized(base + i * vecdim, quantized_query, vecdim, scale, offset);
        
        dis = 1-dis;

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
    return q;
}
