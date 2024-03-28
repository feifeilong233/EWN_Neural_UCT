//
// Created by 25291 on 2024/3/21.
//

#ifndef EWN_BRAVO_CFAKEBOARD_H
#define EWN_BRAVO_CFAKEBOARD_H

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

class FakeBoard {
public:
    double pro[13] = {0.0};
    int LocValue[13] = {0};
    int ChessState[13] = {0};
    double redToBlueOfThread = 0.0;
    double blueToRedOfThread = 0.0;

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

    void getChessState(int b[5][5]) {
        // 遍历board数组
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int a = b[i][j];
                // 如果board上的值是1到12之间的数字，则将对应的ChessState设置为1
                if (a > 0 && a < 13) {
                    ChessState[a] = 1;
                    if (a < 7) {
                        LocValue[a] = RED_VALUE[i][j];
                    } else {
                        LocValue[a] = BLUE_VALUE[i][j];
                    }
                }
            }
        }
        getPro();
        getThread(b);
    }

    double getScore(int col, double beta = 3.48, double lam = 2.18) {
        double expRed = 0;
        double expBlue = 0;

        for (int i = 1; i < 13; ++i) {
            if (ChessState[i] == 0) {
                continue;
            } else {
                if (i < 7) {
                    expRed += pro[i] * LocValue[i];
                } else {
                    expBlue += pro[i] * LocValue[i];
                }
            }
        }

        return col ? lam * (beta * expRed - expBlue) + redToBlueOfThread - blueToRedOfThread :
               lam * (beta * expBlue - expRed) + blueToRedOfThread - redToBlueOfThread;
    }

    void getThread(int a[5][5]) {
        int Direction[3][2] = {{1, 0},
                               {1, 1},
                               {0, 1}};
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                int piece = a[i][j];
                if (piece) {
                    bool f = piece < 7;
                    for (int k = 0; k < 3; k++) {
                        int dx = i + (f ? 1 : -1) * Direction[k][0];
                        int dy = j + (f ? 1 : -1) * Direction[k][1];
                        if (dx >= 0 && dx < 5 && dy >= 0 && dy < 5) {
                            int targetPiece = a[dx][dy];
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
    }

    FakeBoard(int a[5][5]) {
        getChessState(a);
    }
};


#endif //EWN_BRAVO_CFAKEBOARD_H
