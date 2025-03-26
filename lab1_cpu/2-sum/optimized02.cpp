#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

long long recursion(vector<double>& a, int n) {
    if (n == 1) return a[0];
    for (int i = 0; i < n/2; ++i) {
        a[i] += a[n - i - 1];
    }
    return recursion(a, n/2);
}

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

	volatile long long sum = recursion(a, n);

//    cout << "Optimized02 (recursion) sum: " << sum << endl;

    auto end = high_resolution_clock::now(); // 结束计时
    auto duration = duration_cast<microseconds>(end - start).count();

    cout << duration << endl; // 输出执行时间（微秒）
    
    return 0;
}

