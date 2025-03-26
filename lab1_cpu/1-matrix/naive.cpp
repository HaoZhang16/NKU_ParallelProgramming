#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <matrix_size> <input_file>\n";
        return 1;
    }
    
    const int n = stoi(argv[1]);
    ifstream fin(argv[2]);
    
    vector<double> a(n);
    vector<vector<double>> b(n, vector<double>(n));
    vector<double> sum(n, 0.0);

    for (int j = 0; j < n; ++j) fin >> a[j];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            fin >> b[j][i];

    auto start = high_resolution_clock::now(); // 开始计时

    // 平凡算法核心
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            sum[i] += b[j][i] * a[j];  // 列访问
    }

    auto end = high_resolution_clock::now(); // 结束计时
    auto duration = duration_cast<microseconds>(end - start).count();

    cout << duration << endl; // 输出执行时间（微秒）
    
    return 0;
}
