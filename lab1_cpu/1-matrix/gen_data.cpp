#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <matrix_size> <output_file>\n";
        return 1;
    }
    
    const int n = stoi(argv[1]);
    ofstream fout(argv[2]);

    // 生成向量a
    for (int j = 0; j < n; ++j)
        fout << n-j << (j == n-1 ? '\n' : ' ');
    
    // 生成矩阵b（行优先存储）
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i)
            fout << i+j << (i == n-1 ? '\n' : ' ');
    }
    
    return 0;
}

