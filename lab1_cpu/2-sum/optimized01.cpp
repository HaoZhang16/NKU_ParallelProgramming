#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    ifstream infile(argv[1]);
    if (!infile.is_open()) {
        cerr << "Error opening file: " << argv[1] << endl;
        return 1;
    }

    int n;
    infile >> n;

    vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        infile >> a[i];
    }

    infile.close();

    long long sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }

    long long sum = sum1 + sum2;

    cout << "Optimized01 (two-path) sum: " << sum << endl;

    return 0;
}
