#include "ivfpq_openmp.h"
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cstdint>

std::priority_queue<std::pair<float, uint32_t>> ivf_mpi_search(
    float* query,
    float* centroids,
    float* new_base, 
    uint32_t* new_to_old,     
    uint32_t* cluster_start,  
    size_t vecdim,             
    size_t k,                
    size_t n_clusters,        
    size_t m,                  
    int rank,                  // 当前进程 rank
    int world_size             // 总进程数
) {
    // 所有进程本地计算最近的 m 个中心
    std::vector<std::pair<float, uint32_t>> centroid_dists;
    for (size_t i = 0; i < n_clusters; ++i) {
        float* center = centroids + i * vecdim;
        float dis = 1 - InnerProductSIMDNeon(center, query, vecdim);
        centroid_dists.emplace_back(dis, i);
    }

    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + m, centroid_dists.end());
    std::vector<uint32_t> selected_clusters(m);
    for (size_t i = 0; i < m; ++i) {
        selected_clusters[i] = centroid_dists[i].second;
    }

    // 当前 rank 处理其中的部分簇
    std::vector<uint32_t> assigned_clusters;
    for (size_t i = 0; i < m; ++i) {
        if (i % world_size == rank) {
            assigned_clusters.push_back(selected_clusters[i]);
        }
    }

    // 每个进程本地 top-k 优先队列（大顶堆）
    std::priority_queue<std::pair<float, uint32_t>> local_topk;
    size_t num_threads = 1;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_topks(num_threads);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < assigned_clusters.size(); ++i) {
        uint32_t cid = assigned_clusters[i];
        uint32_t begin = cluster_start[cid];
        uint32_t end = cluster_start[cid + 1];
        int tid = omp_get_thread_num();
        auto& plocal_topk = local_topks[tid];

        for (uint32_t j = begin; j < end; ++j) {
            float* base_ptr = new_base + j * vecdim;
            float dis = 1 - InnerProductSIMDNeon(base_ptr, query, vecdim);

            if (plocal_topk.size() < k) {
                plocal_topk.emplace(dis, new_to_old[j]);
            } else if (dis < plocal_topk.top().first) {
                plocal_topk.emplace(dis, new_to_old[j]);
                plocal_topk.pop();
            }
        }
    }

    // 合并线程结果到本地进程结果
    for (auto& q : local_topks) {
        while (!q.empty()) {
            auto p = q.top(); q.pop();
            if (local_topk.size() < k) {
                local_topk.push(p);
            } else if (p.first < local_topk.top().first) {
                local_topk.push(p);
                local_topk.pop();
            }
        }
    }

    // 将 local_topk 转为数组准备收集
    std::vector<std::pair<float, uint32_t>> local_vec;
    while (!local_topk.empty()) {
        local_vec.push_back(local_topk.top());
        local_topk.pop();
    }
    while (local_vec.size() < k) {
        local_vec.emplace_back(1e9f, UINT32_MAX); // 补满
    }

    // root 收集所有结果
    std::vector<std::pair<float, uint32_t>> all_results;
    if (rank == 0) {
        all_results.resize(world_size * k);
    }

    MPI_Gather(
        local_vec.data(), sizeof(std::pair<float, uint32_t>) * k, MPI_BYTE,
        all_results.data(), sizeof(std::pair<float, uint32_t>) * k, MPI_BYTE,
        0, MPI_COMM_WORLD
    );

    std::priority_queue<std::pair<float, uint32_t>> final_topk;
    if (rank == 0) {
        for (const auto& p : all_results) {
            if (final_topk.size() < k) {
                final_topk.push(p);
            } else if (p.first < final_topk.top().first) {
                final_topk.push(p);
                final_topk.pop();
            }
        }
    }

    return final_topk;  // 非 root 进程可返回空堆
}
