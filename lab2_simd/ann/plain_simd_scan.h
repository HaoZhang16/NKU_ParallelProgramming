#include <queue>
#include <arm_neon.h>
#include <fstream>


struct simd8float32 {
    float32x4x2_t data;  // NEON的128位SIMD寄存器，两个4个浮点数的向量

    simd8float32(){
		data.val[0] = vdupq_n_f32(0.0f);
		data.val[1] = vdupq_n_f32(0.0f);
	}

    explicit simd8float32(const float* x) 
        : data{vld1q_f32(x), vld1q_f32(x + 4)} {}

    // 向量乘法
    simd8float32 operator*(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    // 向量加法
    simd8float32 operator+(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    // 将SIMD结果存储到数组
    void storeu(float* output) const {
        vst1q_f32(output, data.val[0]);
        vst1q_f32(output + 4, data.val[1]);
    }
};

float InnerProductSIMDNeon(float* b1, float* b2, size_t vecdim) {
    assert(vecdim % 8 == 0);

    simd8float32 sum;  // 初始化8个浮点数为0
    for (int i = 0; i < vecdim; i += 8) {
        simd8float32 s1(b1 + i), s2(b2 + i);  // 每次加载8个浮点数
        simd8float32 m = s1 * s2;  // 计算内积
        sum = sum + m;  // 累加到总和
    }

    float tmp[8];
    sum.storeu(tmp);  // 存储结果
    float dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return 1 - dis;
}

std::priority_queue<std::pair<float, uint32_t>> plain_simd_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
	
	/*
	// ------------------------------------
	// 估计范围 为SQ准备
	float min_val = std::min(base[0], query[0]);
    float max_val = std::max(base[0], query[0]);

	// 计算 base 数据的最大值和最小值
    for (size_t i = 0; i < base_number; ++i) {
        for (size_t j = 0; j < vecdim; ++j) {
            float val = base[i * vecdim + j];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }

    // 计算 query 数据的最大值和最小值
    for (size_t i = 0; i < vecdim; ++i) {
        float val = query[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }

	// 输出估计的量化区间
    std::cout << "Estimated quantization range: [" << min_val << ", " << max_val << "]\n";

    // 将 min_val 和 max_val 输出到文件 "range.txt"
    std::ofstream outfile("range.txt");  // 打开文件用于写入
    if (outfile.is_open()) {
        outfile << "min_val: " << min_val << "\n";
        outfile << "max_val: " << max_val << "\n";
        outfile.close();  // 关闭文件
        std::cout << "Range written to range.txt\n";
    } else {
        std::cerr << "Error opening file to write range\n";
    }
	// ----------------------------------------------------
	*/

    for (int i = 0; i < base_number; ++i) {
        // 使用SIMD加速点积计算
        float dis = InnerProductSIMDNeon(base + i * vecdim, query, vecdim);
		// dis = 1 - dis;

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

