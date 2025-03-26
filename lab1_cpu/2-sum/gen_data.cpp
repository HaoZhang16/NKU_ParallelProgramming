#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "wrong input\n";
        return 1;
    }
    
    const int n = stoi(argv[1]);
    ofstream fout(argv[2]);

    // 生成向量a
    for (int j = 0; j < n; ++j)
        fout << (n-j)*(j%3) << (j == n-1 ? '\n' : ' ');
    
    return 0;
}


