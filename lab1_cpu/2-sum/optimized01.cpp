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

    // 两路链式
    volatile long long sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }

    volatile long long sum = sum1 + sum2;

//    cout << "Optimized01 (two-path) sum: " << sum << endl;

    auto end = high_resolution_clock::now(); // 结束计时
    auto duration = duration_cast<microseconds>(end - start).count();

    cout << duration << endl; // 输出执行时间（微秒）
    
    return 0;
}
