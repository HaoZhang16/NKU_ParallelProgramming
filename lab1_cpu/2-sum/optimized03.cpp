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

    for (int j = 0; j < n; ++j) fin >> a[j];

    auto start = high_resolution_clock::now(); // 开始计时

    int m = n;
    while (m > 1) {
        m = m / 2;
        for (int i = 0; i < m; ++i) {
            a[i] = a[2*i] + a[2*i+1];
        }
    }

    volatile long long sum = a[0];

//    cout << "Optimized03 (loop) sum: " << sum << endl;

    auto end = high_resolution_clock::now(); // 结束计时
    auto duration = duration_cast<microseconds>(end - start).count();

    cout << duration << endl; // 输出执行时间（微秒）
    
    return 0;
}


