#include <queue>
#include <vector>
#include <cfloat>         // 添加FLT_MAX定义
#include <algorithm>      // 添加swap函数定义
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
        __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 设备交换函数
template <typename T>
__device__ void swap_device(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}

// 内核函数：每个线程块处理一个查询向量
__global__ void flat_batch_search_kernel_nolock(
    const float* base,            // 数据库向量 [n][d]
    const float* query,           // 查询向量 [m][d]
    size_t base_number,           // 数据库数量 n
    size_t query_number,          // 查询数量 m
    size_t vecdim,                // 维度 d
    size_t k,                     // top-k
    float* all_query_distances,   // 输出 [m][k]
    uint32_t* all_query_indices   // 输出 [m][k]
) {
    const size_t qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= query_number) return;

    // 为每个线程分配自己的堆空间
    extern __shared__ char shared[];
    float* local_dist = new float[k];
    uint32_t* local_idx = new uint32_t[k];
    int heap_size = 0;

    const float* qvec = query + qid * vecdim;

    for (size_t i = 0; i < base_number; ++i) {
        const float* bvec = base + i * vecdim;
        float dot = 0.0f;
        for (size_t j = 0; j < vecdim; ++j)
            dot += bvec[j] * qvec[j];

        float dist = 1.0f - dot;

        if (heap_size < k) {
            // 插入
            int pos = heap_size++;
            local_dist[pos] = dist;
            local_idx[pos] = i;
            // 上浮
            while (pos > 0) {
                int parent = (pos - 1) / 2;
                if (local_dist[parent] < local_dist[pos]) {
                    swap_device(local_dist[parent], local_dist[pos]);
                    swap_device(local_idx[parent], local_idx[pos]);
                    pos = parent;
                } else break;
            }
        } else if (dist < local_dist[0]) {
            // 替换堆顶
            local_dist[0] = dist;
            local_idx[0] = i;
            // 下沉
            int pos = 0;
            while (true) {
                int left = 2 * pos + 1;
                int right = 2 * pos + 2;
                int largest = pos;
                if (left < k && local_dist[left] > local_dist[largest]) largest = left;
                if (right < k && local_dist[right] > local_dist[largest]) largest = right;
                if (largest != pos) {
                    swap_device(local_dist[pos], local_dist[largest]);
                    swap_device(local_idx[pos], local_idx[largest]);
                    pos = largest;
                } else break;
            }
        }
    }

    // 写回结果
    size_t offset = qid * k;
    for (size_t i = 0; i < k; ++i) {
        all_query_distances[offset + i] = (i < heap_size) ? local_dist[i] : FLT_MAX;
        all_query_indices[offset + i] = (i < heap_size) ? local_idx[i] : 0;
    }

    delete[] local_dist;
    delete[] local_idx;
}


std::vector<std::priority_queue<std::pair<float, uint32_t>>> flat_search_cuda(
    float* base,           // base[n][d]
    float* query,          // query[m][d]
    size_t base_number,    // n
    size_t query_number,   // m
    size_t vecdim,         // d
    size_t k               // 返回的top-k数量
) {
    // 设备指针
    float *d_base = nullptr, *d_query = nullptr;
    float *d_query_distances = nullptr;
    uint32_t *d_query_indices = nullptr;

    // 分配设备内存
    size_t base_size = base_number * vecdim * sizeof(float);
    size_t query_size = query_number * vecdim * sizeof(float);
    size_t results_dist_size = query_number * k * sizeof(float);
    size_t results_idx_size = query_number * k * sizeof(uint32_t);

    CUDA_CHECK(cudaMalloc(&d_base, base_size));
    CUDA_CHECK(cudaMalloc(&d_query, query_size));
    CUDA_CHECK(cudaMalloc(&d_query_distances, results_dist_size));
    CUDA_CHECK(cudaMalloc(&d_query_indices, results_idx_size));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_base, base, base_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, query_size, cudaMemcpyHostToDevice));

    // 设置线程块和共享内存
    size_t threads_per_block = 64;
    size_t num_blocks = (query_number + threads_per_block - 1) / threads_per_block;

    // 启动改进核函数（不需要共享内存）
    flat_batch_search_kernel_nolock<<<num_blocks, threads_per_block>>>(
        d_base, d_query, base_number, query_number, vecdim, k,
        d_query_distances, d_query_indices
    );


    // 拷贝结果回主机
    std::vector<float> host_query_distances(query_number * k);
    std::vector<uint32_t> host_query_indices(query_number * k);
    CUDA_CHECK(cudaMemcpy(host_query_distances.data(), d_query_distances, 
                         results_dist_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_query_indices.data(), d_query_indices,
                         results_idx_size, cudaMemcpyDeviceToHost));

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_query_distances));
    CUDA_CHECK(cudaFree(d_query_indices));

    // 构建结果优先级队列
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> result(query_number);
    size_t valid_count = std::min(k, base_number); // 有效结果数量

    for (size_t qid = 0; qid < query_number; ++qid) {
        auto& pq = result[qid];
        size_t offset = qid * k;
        // 只添加有效结果
        for (size_t i = 0; i < valid_count; ++i) {
            size_t idx = offset + i;
            pq.emplace(host_query_distances[idx], host_query_indices[idx]);
        }
    }

    return result;
}


// ivf
__global__ void ivf_batch_search_kernel(
    const float* __restrict__ query,
    const float* __restrict__ new_base,
    const uint32_t* __restrict__ new_to_old,
    const uint32_t* __restrict__ cluster_start,
    const uint32_t* __restrict__ selected_clusters,
    size_t vecdim,
    size_t k,
    size_t m,
    float* out_distances,
    uint32_t* out_indices
) {
    extern __shared__ float shmem[]; // 动态共享内存
    float* sh_dist = shmem;
    uint32_t* sh_idx = (uint32_t*)(shmem + blockDim.x * k);

    int qid = blockIdx.x;
    int tid = threadIdx.x;
    const float* qvec = query + qid * vecdim;
    const uint32_t* clusters = selected_clusters + qid * m;

    // 1. 每个线程维护自己的局部top-k
    float local_dist[16];
    uint32_t local_idx[16];
    int local_size = 0;

    for (int mi = 0; mi < m; ++mi) {
        uint32_t cid = clusters[mi];
        uint32_t start = cluster_start[cid];
        uint32_t end = cluster_start[cid + 1];

        for (uint32_t j = start + tid; j < end; j += blockDim.x) {
            const float* base_ptr = new_base + j * vecdim;
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < vecdim; ++d) {
                dot += base_ptr[d] * qvec[d];
            }
            float dist = 1.0f - dot;

            if (local_size < k) {
                local_dist[local_size] = dist;
                local_idx[local_size] = new_to_old[j];
                local_size++;
            } else {
                int max_id = 0;
                for (int t = 1; t < k; ++t) {
                    if (local_dist[t] > local_dist[max_id]) max_id = t;
                }
                if (dist < local_dist[max_id]) {
                    local_dist[max_id] = dist;
                    local_idx[max_id] = new_to_old[j];
                }
            }
        }
    }

    // 2. 将局部结果写入共享内存
    for (int i = 0; i < k; ++i) {
        if (i < local_size) {
            sh_dist[tid * k + i] = local_dist[i];
            sh_idx[tid * k + i] = local_idx[i];
        } else {
            sh_dist[tid * k + i] = FLT_MAX; // 用最大值填充空位
            sh_idx[tid * k + i] = UINT32_MAX;
        }
    }
    __syncthreads();

    // 3. 线程0合并所有线程的结果
    if (tid == 0) {
        float final_dist[16];
        uint32_t final_idx[16];
        int final_size = 0;

        // 初始化最终结果
        for (int i = 0; i < k; ++i) {
            final_dist[i] = FLT_MAX;
        }

        // 遍历所有候选元素 (blockDim.x * k)
        for (int i = 0; i < blockDim.x * k; ++i) {
            float dist = sh_dist[i];
            if (dist == FLT_MAX) continue;

            if (final_size < k) {
                final_dist[final_size] = dist;
                final_idx[final_size] = sh_idx[i];
                final_size++;
            } else {
                // 查找当前结果中的最大距离
                int max_id = 0;
                for (int t = 1; t < k; ++t) {
                    if (final_dist[t] > final_dist[max_id]) max_id = t;
                }
                // 替换最大元素
                if (dist < final_dist[max_id]) {
                    final_dist[max_id] = dist;
                    final_idx[max_id] = sh_idx[i];
                }
            }
        }

        // 4. 写入全局内存
        float* dst_dist = out_distances + qid * k;
        uint32_t* dst_idx = out_indices + qid * k;
        for (int i = 0; i < k; ++i) {
            dst_dist[i] = (i < final_size) ? final_dist[i] : FLT_MAX;
            dst_idx[i] = (i < final_size) ? final_idx[i] : UINT32_MAX;
        }
    }
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_search_cuda(
    float* query,           // [batch][vecdim]
    float* centroids,       // [n_clusters][vecdim]
    float* new_base,        // [N][vecdim]
    uint32_t* new_to_old,   // [N]
    uint32_t* cluster_start,// [n_clusters + 1]
    size_t vecdim,       
    size_t k,             
    size_t n_clusters,     
    size_t m,               
    size_t batch_size     
) {
    std::vector<uint32_t> selected_clusters(batch_size * m);
    for (size_t qid = 0; qid < batch_size; ++qid) {
        std::vector<std::pair<float, uint32_t>> dist_id;
        for (size_t cid = 0; cid < n_clusters; ++cid) {
            float dot = 0.0f;
            for (size_t j = 0; j < vecdim; ++j) {
                dot += query[qid * vecdim + j] * centroids[cid * vecdim + j];
            }
            float dist = 1.0f - dot;
            dist_id.emplace_back(dist, cid);
        }
        std::partial_sort(dist_id.begin(), dist_id.begin() + m, dist_id.end());
        for (size_t i = 0; i < m; ++i) {
            selected_clusters[qid * m + i] = dist_id[i].second;
        }
    }

    // 分配显存
    float *d_query, *d_new_base, *d_out_dist;
    uint32_t *d_new_to_old, *d_cluster_start, *d_selected_clusters, *d_out_idx;
    size_t total_base = cluster_start[n_clusters];

    CUDA_CHECK(cudaMalloc(&d_query, batch_size * vecdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_base, total_base * vecdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_to_old, total_base * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_cluster_start, (n_clusters + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_selected_clusters, batch_size * m * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_dist, batch_size * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_idx, batch_size * k * sizeof(uint32_t)));

    // 拷贝数据到GPU
    CUDA_CHECK(cudaMemcpy(d_query, query, batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new_base, new_base, total_base * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new_to_old, new_to_old, total_base * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_start, cluster_start, (n_clusters + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_selected_clusters, selected_clusters.data(), batch_size * m * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 启动核函数
    size_t threads = 128;
    size_t shared_mem_size = 2 * threads * k * sizeof(float); // 距离+索引
    ivf_batch_search_kernel<<<batch_size, threads, shared_mem_size>>>(
        d_query, d_new_base, d_new_to_old, d_cluster_start, d_selected_clusters,
        vecdim, k, m, d_out_dist, d_out_idx
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷回结果
    std::vector<float> host_dist(batch_size * k);
    std::vector<uint32_t> host_idx(batch_size * k);
    CUDA_CHECK(cudaMemcpy(host_dist.data(), d_out_dist, batch_size * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_idx.data(), d_out_idx, batch_size * k * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 释放显存
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_new_base));
    CUDA_CHECK(cudaFree(d_new_to_old));
    CUDA_CHECK(cudaFree(d_cluster_start));
    CUDA_CHECK(cudaFree(d_selected_clusters));
    CUDA_CHECK(cudaFree(d_out_dist));
    CUDA_CHECK(cudaFree(d_out_idx));

    // 构建结果
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < k; ++j) {
            size_t idx = i * k + j;
            results[i].emplace(host_dist[idx], host_idx[idx]);
        }
    }
    return results;
}

