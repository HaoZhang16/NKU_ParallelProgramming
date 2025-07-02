#include "ivfpq_pthread.h"

void omp_pre_calculate(float* b1, float* b2, float* dist,
                       size_t vecdim, size_t center_num,
                       size_t center_vecdim, size_t cluster_num) {
    #pragma omp parallel for collapse(1) schedule(static)
    for (int i = 0; i < cluster_num; ++i) {
        for (int j = 0; j < center_num; ++j) {
            dist[j + i * center_num] = InnerProductSIMDNeon(
                b1 + (j + i * center_num) * center_vecdim,
                b2 + i * center_vecdim,
                center_vecdim
            );
        }
    }
}

std::priority_queue<std::pair<float, uint32_t>> ivfpq_openmp_search(
    float* query,
    uint8_t* pq_base,
    float* pq_center,
    float* base_full, // 原数据库
    float* ivf_center,
    uint32_t* new_to_old,
    uint32_t* ivf_cluster_start,
    size_t vecdim,
    size_t k,
    size_t pq_center_num,  // 256
    size_t pq_center_vecdim,  // 24
    size_t pq_cluster_num,  // PQ分的段数 4 or 12
    size_t ivf_cluster_num, // 256
    size_t m, // ivf查找的簇数量
    size_t num_threads
) {
    // PQ预处理 可调用PQ_SIMD中的实现
    float* pre_dist = new float[pq_center_num * pq_cluster_num];
    omp_pre_calculate(pq_center, query, pre_dist, vecdim, pq_center_num, pq_center_vecdim, pq_cluster_num);

    // 找出m个簇
    std::vector<std::pair<float, uint32_t>> centroid_dists;
    for (int i = 0; i < ivf_cluster_num; ++i) {
        float* center = ivf_center + i * vecdim;
        float dis = InnerProductSIMDNeon(center, query, vecdim);
        dis = 1 - dis;
        centroid_dists.emplace_back(dis, i);
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + m, centroid_dists.end());
    centroid_dists.resize(m);

    std::vector<PQThreadArg> thread_args(num_threads);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_topks(num_threads);

    size_t rerank = k * 2; // 设置rerank
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int idx = 0; idx < m; ++idx) {
        uint32_t cid = centroid_dists[idx].second;
        float cq_dis = 1 - centroid_dists[idx].first;
        int tid = omp_get_thread_num();
        auto& local_topk = local_topks[tid];

        uint32_t begin = ivf_cluster_start[cid];
        uint32_t end = (cid != ivf_cluster_num-1) ? ivf_cluster_start[cid + 1] : 100000;

        for (uint32_t i = begin; i < end; ++i) {
            double dis = cq_dis;
            // #pragma omp simd reduction(+:dis)
            for (uint32_t j = 0; j < pq_cluster_num; ++j) {
                dis += pre_dist[pq_base[i * pq_cluster_num + j] + j * pq_center_num];
            }
            dis = 1 - dis;

            // 本线程粗排结果
            if (local_topk.size() < rerank) {
                local_topk.emplace(dis, new_to_old[i]);
            } else if (dis < local_topk.top().first) {
                local_topk.emplace(dis, new_to_old[i]);
                local_topk.pop();
            }
        }
    }

    #pragma omp parallel for num_threads(num_threads) schedule(auto)
    for(int idx = 0; idx < (int)num_threads; ++idx){
        int tid = omp_get_thread_num();
        auto& local_topk = local_topks[tid];
        // 对粗排结果全精度重排
        std::priority_queue<std::pair<float, uint32_t>> precise_heap;
        while (!local_topk.empty()) {
            auto top_pair = local_topk.top();
            uint32_t idx = top_pair.second;
            local_topk.pop();

            // 计算真实距离
            float true_dis = InnerProductSIMDNeon(base_full + idx * vecdim, query, vecdim);
            true_dis = 1 - true_dis;

            if (precise_heap.size() < k) {
                precise_heap.emplace(true_dis, idx);
            } else if (true_dis < precise_heap.top().first) {
                precise_heap.emplace(true_dis, idx);
                precise_heap.pop();
            }
        }

        // 把重排后的 k 个放回 local_topk，方便主线程合并
        local_topk = std::move(precise_heap);
    }

    // 合并 top-k
    std::priority_queue<std::pair<float, uint32_t>> final_topk;
    for (int i = 0; i < (int)num_threads; ++i) {
        auto& local_q = local_topks[i];
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
