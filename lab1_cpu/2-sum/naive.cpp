#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "wrong input\n";
        return 1;
    }
    
    const int n = stoi(argv[1]);
    ifstream fin(argv[2]);
    
    vector<double> a(n);
    for (int j = 0; j < n; ++j) fin >> a[j];

    auto start = high_resolution_clock::now(); // 开始计时

    // 平凡算法核心
    volatile long long sum = 0; // 避免被编译器优化
    for (int i = 0; i < n; ++i) {
        sum += a[i];
    }

//    cout << "Naive sum: " << sum << endl;

    auto end = high_resolution_clock::now(); // 结束计时
    auto duration = duration_cast<microseconds>(end - start).count();

    cout << duration << endl; // 输出执行时间（微秒）
    
    return 0;
}
