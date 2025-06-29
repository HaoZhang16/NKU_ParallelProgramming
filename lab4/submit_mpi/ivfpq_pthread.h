#include "ivf_openmp.h"

struct PQThreadArg {
    float* query;
    float* base_full;  // 原数据库
    uint8_t* new_base; // 100000*4
    uint32_t* new_to_old;
    uint32_t* cluster_start;
    float* pre_dist; // PQ预处理距离
    size_t pq_cluster_num; // PQ分段数
    size_t pq_center_num; // PQ每段聚类数
    std::vector<std::pair<float, uint32_t>> cluster_dist; // 该线程负责的簇距离+簇编号
    size_t vecdim;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t>> local_topk;
    bool isPQIVF;
};

void* PQ_search_thread_func(void* arg_void) {
    PQThreadArg* arg = (PQThreadArg*)arg_void;

    size_t rerank = arg->k * 15; // 设置rerank
    for (auto& cid : arg->cluster_dist) {
        uint32_t begin = arg->cluster_start[cid.second];
        uint32_t end = arg->cluster_start[cid.second + 1];
        double cq_dis = 1 - cid.first;

        for (uint32_t i = begin; i < end; ++i) {
            float dis = arg->isPQIVF ? 0 : cq_dis;
            for(uint32_t j = 0; j < arg->pq_cluster_num; ++j){
                dis += arg->pre_dist[arg->new_base[i * arg->pq_cluster_num + j] + j * arg->pq_center_num];
            }
            dis = 1 - dis;

            // 本线程粗排结果
            if (arg->local_topk.size() < rerank) {
                arg->local_topk.emplace(dis, arg->new_to_old[i]);
            } else if (dis < arg->local_topk.top().first) {
                arg->local_topk.emplace(dis, arg->new_to_old[i]);
                arg->local_topk.pop();
            }
        }
    }

    // 对粗排结果全精度重排
    std::priority_queue<std::pair<float, uint32_t>> precise_heap;
    while (!arg->local_topk.empty()) {
        auto top_pair = arg->local_topk.top();
        uint32_t idx = top_pair.second;
        arg->local_topk.pop();

        // 计算真实距离
        float true_dis = InnerProductSIMDNeon(arg->base_full + idx * arg->vecdim, arg->query, arg->vecdim);
        true_dis = 1 - true_dis;

        if (precise_heap.size() < arg->k) {
            precise_heap.emplace(true_dis, idx);
        } else if (true_dis < precise_heap.top().first) {
            precise_heap.emplace(true_dis, idx);
            precise_heap.pop();
        }
    }

    // 把重排后的 k 个放回 local_topk，方便主线程合并
    arg->local_topk = std::move(precise_heap);

    return nullptr;
}

std::priority_queue<std::pair<float, uint32_t>> ivfpq_pthread_search(
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
    size_t pq_cluster_num,  // PQ分的段数 4 这里cluster和center混了ToT 依然沿用SIMD的名称
    size_t ivf_cluster_num, // 256
    size_t m, // ivf查找的簇数量
    size_t num_threads
){
    // PQ预处理 可调用PQ_SIMD中的实现
    float* pre_dist = new float[pq_center_num * pq_cluster_num];
    pre_calculate(pq_center, query, pre_dist, vecdim, pq_center_num, pq_center_vecdim, pq_cluster_num);

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

    // std::vector<uint32_t> selected_clusters;
    // for (int i = 0; i < m; ++i) selected_clusters.push_back(centroid_dists[i].second);

    // 分配任务
    std::vector<std::vector<std::pair<float, uint32_t>>> thread_tasks(num_threads);
    for (size_t i = 0; i < centroid_dists.size(); ++i) {
        thread_tasks[i % num_threads].push_back(centroid_dists[i]);
    }

    std::vector<PQThreadArg> thread_args(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        thread_args[i] = PQThreadArg{
            .query = query,
            .base_full = base_full,
            .new_base = pq_base,
            .new_to_old = new_to_old,
            .cluster_start = ivf_cluster_start,
            .pre_dist = pre_dist,
            .pq_cluster_num = pq_cluster_num,
            .pq_center_num = pq_center_num,
            .cluster_dist = thread_tasks[i],
            .vecdim = vecdim,
            .k = k,
            .isPQIVF = false
        };
        pthread_create(&threads[i], nullptr, PQ_search_thread_func, &thread_args[i]);
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

// 先PQ再IVF
std::priority_queue<std::pair<float, uint32_t>> pqivf_pthread_search(
    float* query, 
    uint8_t* pq_base, 
    float* pq_center, 
    float* base_full, // 原数据库
    uint8_t* ivf_center,  // IVF聚类中心向量是4维的uint8
    uint32_t* new_to_old,
    uint32_t* ivf_cluster_start,
    size_t vecdim,
    size_t k, 
    size_t pq_center_num,  // 256
    size_t pq_center_vecdim,  // 24
    size_t pq_cluster_num,  // PQ分的段数 4 这里cluster和center混了ToT 依然沿用SIMD的名称
    size_t ivf_cluster_num, // 256
    size_t m, // ivf查找的簇数量
    size_t num_threads
){
    // PQ预处理 可调用PQ_SIMD中的实现
    float* pre_dist = new float[pq_center_num * pq_cluster_num];
    pre_calculate(pq_center, query, pre_dist, vecdim, pq_center_num, pq_center_vecdim, pq_cluster_num);

    // 找出m个簇
    std::vector<std::pair<float, uint32_t>> centroid_dists;
    for (int i = 0; i < ivf_cluster_num; ++i) {
        uint8_t* center = ivf_center + i * pq_cluster_num;
        float dis = 0;
        for(int j = 0; j < pq_cluster_num; ++j)
            dis += pre_dist[center[j] + j * pq_center_num];
        dis = 1 - dis;
        centroid_dists.emplace_back(dis, i);
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + m, centroid_dists.end());
    centroid_dists.resize(m);

    // 分配任务
    std::vector<std::vector<std::pair<float, uint32_t>>> thread_tasks(num_threads);
    for (size_t i = 0; i < centroid_dists.size(); ++i) {
        thread_tasks[i % num_threads].push_back(centroid_dists[i]);
    }

    std::vector<PQThreadArg> thread_args(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        thread_args[i] = PQThreadArg{
            .query = query,
            .base_full = base_full,
            .new_base = pq_base,
            .new_to_old = new_to_old,
            .cluster_start = ivf_cluster_start,
            .pre_dist = pre_dist,
            .pq_cluster_num = pq_cluster_num,
            .pq_center_num = pq_center_num,
            .cluster_dist = thread_tasks[i],
            .vecdim = vecdim,
            .k = k,
            .isPQIVF = true
        };
        pthread_create(&threads[i], nullptr, PQ_search_thread_func, &thread_args[i]);
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