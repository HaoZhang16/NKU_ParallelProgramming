#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

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

    int* tmp = new int[n];
    memcpy(tmp, a.data(), n * sizeof(int));

    int m = n;
    while (m > 1) {
        int new_m = m / 2;
        for (int i = 0; i < new_m; ++i) {
            tmp[i] = tmp[2*i] + tmp[2*i+1];
        }
        m = new_m;
    }

    long long sum = tmp[0];
    delete[] tmp;

    cout << "Optimized03 (loop) sum: " << sum << endl;

    return 0;
}
