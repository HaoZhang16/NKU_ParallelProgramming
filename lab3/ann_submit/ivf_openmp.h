#include <omp.h>
#include <utility>
#include <algorithm>
#include <cstdint>
#include "ivf_pthread.h"

std::priority_queue<std::pair<float, uint32_t>> ivf_openmp_search(
    float* query,
    float* centroids,
    float* new_base,
    uint32_t* new_to_old,
    uint32_t* cluster_start,
    size_t vecdim,
    size_t k,
    size_t n_clusters,
    size_t m,
    size_t num_threads
) {
    // 找出m个簇
    std::vector<std::pair<float, uint32_t>> centroid_dists;
    for (size_t i = 0; i < n_clusters; ++i) {
        float* center = centroids + i * vecdim;
        float dis = 1 - InnerProductSIMDNeon(center, query, vecdim);
        centroid_dists.emplace_back(dis, i);
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + m, centroid_dists.end());

    std::vector<uint32_t> selected_clusters(m);
    for (size_t i = 0; i < m; ++i) selected_clusters[i] = centroid_dists[i].second;

    // 分配任务
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_topks(num_threads);

    // 并行处理 selected_clusters 中的每个簇
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < m; ++i) {
        uint32_t cid = selected_clusters[i];
        uint32_t begin = cluster_start[cid];
        uint32_t end = cluster_start[cid + 1];
        int tid = omp_get_thread_num();
        auto& local_topk = local_topks[tid];

        for (uint32_t j = begin; j < end; ++j) {
            float* base_ptr = new_base + j * vecdim;
            float dis = 1 - InnerProductSIMDNeon(base_ptr, query, vecdim);

            if (local_topk.size() < k) {
                local_topk.emplace(dis, new_to_old[j]);
            } else if (dis < local_topk.top().first) {
                local_topk.emplace(dis, new_to_old[j]);
                local_topk.pop();
            }
        }
    }

    // 合并 top-k
    std::priority_queue<std::pair<float, uint32_t>> final_topk;
    for (auto& local_q : local_topks) {
        while (!local_q.empty()) {
            auto entry = local_q.top(); local_q.pop();
            if (final_topk.size() < k) {
                final_topk.push(entry);
            } else if (entry.first < final_topk.top().first) {
                final_topk.push(entry);
                final_topk.pop();
            }
        }
    }

    return final_topk;
}
