#pragma once
#include <vector>
#include <queue>

std::vector<std::priority_queue<std::pair<float, uint32_t>>> flat_search_cuda(
    float* base, float* queries,
    size_t n, size_t m, size_t d, size_t k
);

std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_search_cuda(
    float* query,           // [batch][vecdim]
    float* centroids,       // [n_clusters][vecdim]
    float* new_base,        // [N][vecdim]
    uint32_t* new_to_old,   // [N]
    uint32_t* cluster_start,// [n_clusters + 1]
    size_t vecdim,          // 向量维度
    size_t k,               // top-k
    size_t n_clusters,      // 聚类中心数量
    size_t m,               // nprobe（每个query选择m个簇）
    size_t batch_size       // 查询向量数量
);
