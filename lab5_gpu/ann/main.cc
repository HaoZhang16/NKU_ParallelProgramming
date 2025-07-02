#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "batch_search.h"
#include "cuda_batch_search.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./files/"; 
    std::string q_data_path = "./files/";  // 本地测试
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    auto sq_base = LoadData<uint8_t>(q_data_path + "DEEP100K.base.100k.ubin", base_number, vecdim);

    size_t center_vecdim = 0, center_num_total = 0;
    size_t center_num = 0, cluster_num = 0;
    auto pq_base = LoadData<uint8_t>(q_data_path + "DEEP100K.base.100k_4_256.quantized.bin", base_number, cluster_num);
    auto pq_center = LoadData<float>(q_data_path + "DEEP100K.base.100k_4_256.center.bin", center_num_total, center_vecdim);
    center_num = center_num_total / cluster_num;

    size_t fs_center_num = 0;
    auto fs_base = LoadData<uint8_t>(q_data_path + "DEEP100K.base.100k_4_16.quantized.bin", base_number, cluster_num);
    auto fs_center = LoadData<float>(q_data_path + "DEEP100K.base.100k_4_16.center.bin", fs_center_num, center_vecdim);
    fs_center_num /= 4;
    
    // ivf相关数据，和ivfpq共用
    size_t ivf_n_clusters = 0, idx_size = 0, offset_num = 0;
    auto ivf_center = LoadData<float>(q_data_path + "DEEP100K.base.100k.256.center.bin", ivf_n_clusters, vecdim);//256*96
    auto ivf_data = LoadData<float>(q_data_path + "DEEP100K.base.100k.256.data.bin", base_number, vecdim);//100000*96
    auto ivf_index = LoadData<uint32_t>(q_data_path + "DEEP100K.base.100k.256.index.bin", base_number, idx_size);//100000*1
    auto ivf_offset_raw = LoadData<uint32_t>(q_data_path + "DEEP100K.base.100k.256.offset.bin", offset_num, idx_size);//256*1
	uint32_t* ivf_offset = new uint32_t[offset_num+1];
	memcpy(ivf_offset, ivf_offset_raw, offset_num * sizeof(uint32_t));
	ivf_offset[offset_num] = base_number;

    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    
    // 查询测试代码
    size_t batch_size = 2048; 

    for(int i = 0; i < test_number; i += batch_size) {
        size_t actual_batch = std::min(batch_size, test_number - i); // 确保能取够
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
		// 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        
		// 朴素
		// auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
        
        // ivf-omp
        // auto res = ivf_openmp_search(test_query + i*vecdim, ivf_center, ivf_data, ivf_index, ivf_offset, vecdim, k, ivf_n_clusters, 8, 8);

        // cpu-batch
        // auto res = flat_batch_search(base, test_query + i * vecdim, base_number, actual_batch, vecdim, k);

        // gpu-cuda
		// std::vector<std::priority_queue<std::pair<float, uint32_t>>> res(actual_batch);
		if(i == 0) std::cout<<"begin\n";
		// auto res = flat_search_cuda(base, test_query + i * vecdim, base_number, actual_batch, vecdim, k);
		// if(i == 0) std::cout<<"end\n";

		// gpu-ivf
		auto res = ivf_search_cuda(test_query + i * vecdim, ivf_center, ivf_data, ivf_index, ivf_offset, vecdim, k, ivf_n_clusters, 256, actual_batch); 
		if(i == 0) std::cout<<"end of batch1\n";

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        for (size_t j = 0; j < actual_batch; ++j) {
            std::set<uint32_t> gtset;
            for (int l = 0; l < k; ++l) {
                gtset.insert(test_gt[(i + j) * test_gt_d + l]);
            }

            size_t acc = 0;
            auto& pq = res[j];
            while (!pq.empty()) {
                if (gtset.find(pq.top().second) != gtset.end()) ++acc;
                pq.pop();
            }
            float recall = (float)acc / k;
            results[i + j] = {recall, static_cast<int64_t>(diff / actual_batch)};
        }
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}

