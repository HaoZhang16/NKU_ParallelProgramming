#include <pthread.h>
#include "fs_simd_scan.h"

struct ThreadArg {
    float* query;
    float* new_base;
    uint32_t* new_to_old;
    uint32_t* cluster_start;
    std::vector<uint32_t> cluster_ids; // 该线程负责的簇编号
    size_t vecdim;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t>> local_topk;
};

void* search_thread_func(void* arg_void) {
    ThreadArg* arg = (ThreadArg*)arg_void;

    for (uint32_t cid : arg->cluster_ids) {
        uint32_t begin = arg->cluster_start[cid];
        uint32_t end = arg->cluster_start[cid + 1];

        for (uint32_t i = begin; i < end; ++i) {
            float* base_ptr = arg->new_base + i * arg->vecdim;
            float dis = InnerProductSIMDNeon(base_ptr, arg->query, arg->vecdim);
            dis = 1 - dis;

            if (arg->local_topk.size() < arg->k) {
                arg->local_topk.emplace(dis, arg->new_to_old[i]);
            } else if (dis < arg->local_topk.top().first) {
                arg->local_topk.emplace(dis, arg->new_to_old[i]);
                arg->local_topk.pop();
            }
        }
    }

    return nullptr;
}

std::priority_queue<std::pair<float, uint32_t>> ivf_pthread_search(
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
    for (int i = 0; i < n_clusters; ++i) {
        float* center = centroids + i * vecdim;
        float dis = InnerProductSIMDNeon(center, query, vecdim);
        dis = 1 - dis;
        centroid_dists.emplace_back(dis, i);
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + m, centroid_dists.end());
    std::vector<uint32_t> selected_clusters;
    for (int i = 0; i < m; ++i) selected_clusters.push_back(centroid_dists[i].second);

    // 分配任务
    std::vector<std::vector<uint32_t>> thread_tasks(num_threads);
    for (size_t i = 0; i < selected_clusters.size(); ++i) {
        thread_tasks[i % num_threads].push_back(selected_clusters[i]);
    }

    std::vector<ThreadArg> thread_args(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        thread_args[i] = ThreadArg{
            .query = query,
            .new_base = new_base,
            .new_to_old = new_to_old,
            .cluster_start = cluster_start,
            .cluster_ids = thread_tasks[i],
            .vecdim = vecdim,
            .k = k
        };
        pthread_create(&threads[i], nullptr, search_thread_func, &thread_args[i]);
    }

    for (size_t i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // 合并 top-k
    std::priority_queue<std::pair<float, uint32_t>> final_topk;

    for (int i = 0; i < num_threads; ++i) {
        auto& local_q = thread_args[i].local_topk;
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
