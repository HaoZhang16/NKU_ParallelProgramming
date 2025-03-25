#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>  // for atoi

using namespace std;

int main(int argc, char* argv[]) {
    vector<int> test_ns;
    
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            test_ns.push_back(atoi(argv[i]));
        }
    } else {
        test_ns = {8, 16, 32, 64};
    }

    for (int n : test_ns) {
        string filename = "data_n" + to_string(n) + ".txt";
        ofstream outfile(filename);
        
        outfile << n << endl;
        for (int i = 1; i <= n; ++i) {
            outfile << i << endl;
        }
        
        outfile.close();
    }

    return 0;
}
