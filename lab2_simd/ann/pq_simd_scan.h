#include <queue>
#include <arm_neon.h>
#include "plain_simd_scan.h"

// 利用InnerProductSIMDNeon进行24个float32运算
// b1是所有中心向量，b2是查询向量，center_num表示每段的聚类中心数量
void pre_calculate(float* b1, float* b2, float* dist, size_t vecdim, size_t center_num, size_t center_vecdim, size_t cluster_num){
    for(int i = 0; i < cluster_num ; ++i){ // 遍历每段
        for(int j=0; j < center_num ; ++j){
            dist[j + i*center_num] = InnerProductSIMDNeon(b1 + j*center_vecdim + i*center_num*center_vecdim, b2 + i*center_vecdim, center_vecdim);
        }
    }
}

std::priority_queue<std::pair<float, uint32_t>> pq_simd_search(uint8_t* base, float* center, float* query, size_t base_number, size_t vecdim,
     size_t k, size_t center_num, size_t center_vecdim, size_t cluster_num) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 预处理
    float* pre_dist = new float[center_num * cluster_num];
    pre_calculate(center, query, pre_dist, vecdim, center_num, center_vecdim, cluster_num);

    for (int i = 0; i < base_number; ++i) {
        // 使用预处理结果
        float dis = 0;
        for(int j=0;j<cluster_num;++j){
            dis += pre_dist[base[i*cluster_num + j] + center_num * j]; // base在i*cluster_num + j的值即为聚类中心编号(0-center_num-1)
        }
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
    delete[] pre_dist;
    return q;
}