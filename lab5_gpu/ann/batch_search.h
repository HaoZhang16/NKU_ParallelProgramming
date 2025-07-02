#include <queue>

std::vector<std::priority_queue<std::pair<float, uint32_t>>> flat_batch_search(
    float* base,           // base[n][d]
    float* query,          // query[m][d]
    size_t base_number,    // n
    size_t query_number,   // m
    size_t vecdim,         // d
    size_t k
) {
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> result(query_number);

    for (size_t i = 0; i < base_number; ++i) {
        for (size_t qid = 0; qid < query_number; ++qid) {
            float dis = 0.0f;
            for (size_t d = 0; d < vecdim; ++d) {
                dis += base[i * vecdim + d] * query[qid * vecdim + d];
            }
            dis = 1.0f - dis;

            auto& q = result[qid];
            if (q.size() < k) {
                q.emplace(dis, i);
            } else if (dis < q.top().first) {
                q.emplace(dis, i);
                q.pop();
            }
        }
    }
    return result;
}
