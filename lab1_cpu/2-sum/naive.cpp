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

    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += a[i];
    }

    cout << "Naive sum: " << sum << endl;

    return 0;
}
