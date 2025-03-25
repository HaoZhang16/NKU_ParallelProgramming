#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

using namespace std;

long long recursion(int* a, int n) {
    if (n == 1) return a[0];
    for (int i = 0; i < n/2; ++i) {
        a[i] += a[n - i - 1];
    }
    return recursion(a, n/2);
}

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

    long long sum = recursion(tmp, n);
    delete[] tmp;

    cout << "Optimized02 (recursive) sum: " << sum << endl;

    return 0;
}
