/*
* 2020年5月
* 
*匈牙利匹配算法，是一种典型的一对一的配对算法。一对一的匹配，与之对应的，
*就是一对多，或多对一的配对算法。也称作 no anchor 操作
*/
#include "hungarianoper.h"
//sisyphus
Eigen::Matrix<float, -1, 2, Eigen::RowMajor> HungarianOper::Solve(const DYNAMICM &cost_matrix)
{
    int rows = cost_matrix.rows();
    int cols = cost_matrix.cols();
    Matrix<double> matrix(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            matrix(row, col) = cost_matrix(row, col);
        }
    }
    //Munkres get matrix;
    Munkres<double> m;
    m.solve(matrix);

    //
    std::vector<std::pair<int, int>> pairs;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int tmp = (int)matrix(row, col);
            if (tmp == 0) pairs.push_back(std::make_pair(row, col));
        }
    }
    //
    int count = pairs.size();
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> re(count, 2);
    for (int i = 0; i < count; i++) {
        re(i, 0) = pairs[i].first;
        re(i, 1) = pairs[i].second;
    }
    return re;
}//end Solve;
