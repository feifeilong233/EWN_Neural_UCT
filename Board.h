//
// Created by yakumohakuryu on 2024/4/8.
//

#ifndef EWN_BRAVO_BOARD_H
#define EWN_BRAVO_BOARD_H

static int BLUE_VALUE[5][5] = {{50, 10, 6, 3, 1},
                               {10, 8,  4, 2, 1},
                               {6,  4,  4, 2, 1},
                               {3,  2,  2, 2, 1},
                               {1,  1,  1, 1, 0}};

static int RED_VALUE[5][5] = {{0, 1, 1, 1,  1},
                              {1, 2, 2, 2,  3},
                              {1, 2, 4, 4,  6},
                              {1, 2, 4, 8,  10},
                              {1, 3, 6, 10, 50}};

const double lamda = 5;
const double betaone = 2.2;

class Board {
public:
    std::shared_ptr<Board> parent;
    std::vector<int> validchess;            //可以走的棋子
    int chess[2] = {0, -1};            //导致这个棋局要走的棋子和方向
    std::vector<int> posStep[2];            //可能的走法的个数，要分种类，0水平走位、1垂直走位、2对角走位
    int color;                    //红色为0，蓝色为1
    int board[5][5] = {};                //棋盘
    int visit_times = 0;
    int win_time = 0;
    double quality = 0.0;
    double pro[13] = {0.0};            //棋子移动概率
    int LocValue[13] = {0};
    int ChessState[13] = {0};            //棋子状态
    std::vector<std::shared_ptr<Board>> child;

    // 递归地统计以当前节点为根节点的子树的深度
    int subtreeDepth() const {
        int maxDepth = 0;
        for (const auto& childNode : child) {
            int childDepth = childNode->subtreeDepth();
            if (childDepth > maxDepth) {
                maxDepth = childDepth;
            }
        }
        return maxDepth + 1; // 加1表示当前节点的深度
    }

    std::pair<int, int> findNearby(int n) {
        std::pair<int, int> ans = {-1, -1};
        if (n > 6) {
            for (int i = n - 1; i >= 7; --i) {
                if (ChessState[i] != 0) {
                    ans.first = i;
                    break;
                }
            }
            for (int i = n + 1; i <= 12; ++i) {
                if (ChessState[i] != 0) {
                    ans.second = i;
                    break;
                }
            }
        } else {
            for (int i = n - 1; i >= 1; --i) {
                if (ChessState[i] != 0) {
                    ans.first = i;
                    break;
                }
            }
            for (int i = n + 1; i <= 6; ++i) {
                if (ChessState[i] != 0) {
                    ans.second = i;
                    break;
                }
            }
        }
        return ans;
    }

    void getPro() {
        std::fill_n(pro, 13, 1.0 / 6);
        pro[0] = 0;
        //计算棋子移动概率（结果×6）
        for (int i = 1; i < 13; i++) {
            if (ChessState[i] == 0) {
                int pr, pl;
                std::tie(pr, pl) = findNearby(i);
                if (pr != -1 && pl != -1) {
                    if (LocValue[pr] > LocValue[pl]) {
                        pro[pr] += pro[i];
                    } else if (LocValue[pr] == LocValue[pl]) {
                        pro[pr] += pro[i] / 2;
                        pro[pl] += pro[i] / 2;
                    } else {
                        pro[pl] += pro[i];
                    }
                } else if (pr != -1) {
                    pro[pr] += pro[i];
                } else if (pl != -1) {
                    pro[pl] += pro[i];
                }
                pro[i] = 0;
            }
        }
    }

    double getScore(double beta = betaone, double lam = lamda) {
        double redToBlueOfThread, blueToRedOfThread;
        std::tie(redToBlueOfThread, blueToRedOfThread) = getThread();
        double expRed = 0;
        double expBlue = 0;

        for (int i = 1; i < 13; ++i) {
            if (i < 7) {
                if (ChessState[i] == 0) {
                    continue;
                }
                expRed += pro[i] * LocValue[i];
            } else {
                if (ChessState[i] == 0) {
                    continue;
                }
                expBlue += pro[i] * LocValue[i];
            }
        }

        double theValue = color ? lam * (beta * expRed - expBlue) + redToBlueOfThread - blueToRedOfThread :
                          lam * (beta * expBlue - expRed) + blueToRedOfThread - redToBlueOfThread;
        return theValue;
    }

    std::pair<double, double> getThread() {
        double redToBlueOfThread = 0.0;
        double blueToRedOfThread = 0.0;

        // 计算威胁值
        int Direction[3][2] = {{1, 0},
                               {1, 1},
                               {0, 1}};
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                int piece = board[i][j];
                if (piece) {
                    bool f = piece < 7;
                    for (int k = 0; k < 3; k++) {
                        int dx = i + (f ? 1 : -1) * Direction[k][0];
                        int dy = j + (f ? 1 : -1) * Direction[k][1];
                        if (dx >= 0 && dx < 5 && dy >= 0 && dy < 5) {
                            int targetPiece = board[dx][dy];
                            if (targetPiece && ((f && targetPiece > 6) || (!f && targetPiece < 7))) {
                                if (f) {
                                    redToBlueOfThread += pro[piece];
                                } else {
                                    blueToRedOfThread += pro[piece];
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
        return std::make_pair(redToBlueOfThread, blueToRedOfThread);
    }

    Board(std::shared_ptr<Board> par, int col, int b[5][5], int pos[2]) {
        parent = par;
        color = col;
        chess[0] = pos[0];
        chess[1] = pos[1];
        // 遍历board数组
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int a = b[i][j];
                board[i][j] = a;
                // 如果board上的值是1到12之间的数字，则将对应的ChessState设置为1
                if (a > 0 && a < 13) {
                    ChessState[a] = 1;
                    if (a < 7) {
                        LocValue[a] = RED_VALUE[i][j];
                        if (col == 0)
                            validchess.push_back(a);
                    } else {
                        LocValue[a] = BLUE_VALUE[i][j];
                        if (col == 1)
                            validchess.push_back(a - 6);
                    }
                }
            }
        }
        getPro();
        com_posStep();
    }

    void com_posStep() {
        if (color == 0) {
            for (int i = 0; i < validchess.size(); i++) {
                int x, y;
                for (int j = 0; j < 5; j++)
                    for (int k = 0; k < 5; k++) {
                        if (board[j][k] == validchess[i]) {
                            x = j;
                            y = k;
                            break;
                        }
                    }
                if (x == 4) {
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(0);
                } else if (y == 4) {
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(1);
                } else {
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(0);
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(1);
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(2);
                }
            }
        } else {
            for (int i = 0; i < validchess.size(); i++) {
                int x, y;
                for (int j = 0; j < 5; j++)
                    for (int k = 0; k < 5; k++) {
                        if (board[j][k] == validchess[i] + 6) {
                            x = j;
                            y = k;
                            break;
                        }
                    }
                if (x == 0) {
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(0);
                } else if (y == 0) {
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(1);
                } else {
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(0);
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(1);
                    posStep[0].push_back(validchess[i]);
                    posStep[1].push_back(2);
                }
            }
        }
    }

    Board(std::shared_ptr<Board> par, int col, int a[5][5], int dice) {
        parent = par;
        color = col;
        // 遍历board数组
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int temp = a[i][j];
                board[i][j] = temp;
                // 如果board上的值是1到12之间的数字，则将对应的ChessState设置为1
                if (temp > 0 && temp < 13) {
                    ChessState[temp] = 1;
                }
            }
        }

        if (col == 0) {
            if (ChessState[dice] == 1)
                validchess.push_back(dice);
            else {
                for (int i = dice - 1; i > 0; i--) {
                    if (ChessState[i] == 1) {
                        validchess.push_back(i);
                        break;
                    }
                }
                for (int i = dice + 1; i < 7; i++) {
                    if (ChessState[i] == 1) {
                        validchess.push_back(i);
                        break;
                    }
                }
            }
        } else {
            if (ChessState[dice] == 1)
                validchess.push_back(dice - 6);
            else {
                for (int i = dice - 1; i > 6; i--) {
                    if (ChessState[i] == 1) {
                        validchess.push_back(i - 6);
                        break;
                    }
                }
                for (int i = dice + 1; i < 13; i++) {
                    if (ChessState[i] == 1) {
                        validchess.push_back(i - 6);
                        break;
                    }
                }
            }
        }
        com_posStep();
    }
};

#endif //EWN_BRAVO_BOARD_H
