#include <queue>
#include <arm_neon.h>
#include <fstream>


struct simd8float32 {
    float32x4x2_t data;  // NEON的128位SIMD寄存器，两个4个浮点数的向量

    simd8float32(){
		data.val[0] = vdupq_n_f32(0.0f);
		data.val[1] = vdupq_n_f32(0.0f);
	}

    explicit simd8float32(const float x){
        data.val[0] = vdupq_n_f32(x);
        data.val[1] = vdupq_n_f32(x);
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

    // 向量减法
    simd8float32 operator-(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vsubq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vsubq_f32(data.val[1], other.data.val[1]);
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

    // float tmp[8];
    // sum.storeu(tmp);  // 存储结果
    // float dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    float32x4_t part0 = sum.data.val[0];
    float32x4_t part1 = sum.data.val[1];
    float dis = vaddvq_f32(part0) + vaddvq_f32(part1);  // 各自内部 4 加

    return dis;
}

std::priority_queue<std::pair<float, uint32_t>> plain_simd_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < base_number; ++i) {
        // 使用SIMD加速点积计算
        float dis = InnerProductSIMDNeon(base + i * vecdim, query, vecdim);
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


