#include "CEinstein.h"

time_t now_time;
time_t begin_time;
time_t end_time;
struct tm *pt;

const double coe = 1.414;
const double kilo = 10;
const double tempo = 1;
std::uniform_int_distribution<int> dis(1, 6);
std::random_device rd;
std::mt19937 generator(rd());

torch::DeviceType device_type = at::kCPU;
torch::Device device(device_type);
torch::jit::script::Module model = torch::jit::load("Alpha1.pt", device);
torch::NoGradGuard no_grad;

// 获取红蓝方的5x5map
void get_alldata_5(std::shared_ptr<Board> current_status, std::deque<std::array<std::array<int, 5>, 5>> &DQ) {
    // 处理当前状态的board
    std::array<std::array<int, 5>, 5> datanew1{};
    std::array<std::array<int, 5>, 5> datanew2{};
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            int temp = current_status->board[i][j];
            if (temp && temp < 7) {
                datanew1[i][j] = temp;
            } else if (temp > 6) {
                datanew2[i][j] = temp;
            }
        }
    }
    DQ.push_back(datanew1);
    DQ.push_back(datanew2);
    // 处理父状态的board
    auto parent_status = current_status->parent;
    for (int ii = 0; ii < 4; ++ii) {
        if (parent_status) {
            std::array<std::array<int, 5>, 5> datanew3{};
            std::array<std::array<int, 5>, 5> datanew4{};
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    int temp = parent_status->board[i][j];
                    if (temp && temp < 7) {
                        datanew3[i][j] = temp;
                    } else if (temp > 6) {
                        datanew4[i][j] = temp;
                    }
                }
            }
            DQ.push_back(datanew3);
            DQ.push_back(datanew4);
            parent_status = parent_status->parent;
        } else {
            break;
        }
    }
}

void get_alldata_5blue(std::shared_ptr<Board> current_status, std::deque<std::array<std::array<int, 5>, 5>> &DQ) {
    // 处理当前状态的board
    std::array<std::array<int, 5>, 5> datanew1{};
    std::array<std::array<int, 5>, 5> datanew2{};
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            int temp = current_status->board[i][j];
            if (temp && temp < 7) {
                datanew1[j][i] = temp + 6;
            } else if (temp > 6) {
                datanew2[j][i] = temp - 6;
            }
        }
    }
    DQ.push_back(datanew2);
    DQ.push_back(datanew1);
    // 处理父状态的board
    auto parent_status = current_status->parent;
    for (int ii = 0; ii < 4; ++ii) {
        if (parent_status) {
            std::array<std::array<int, 5>, 5> datanew3{};
            std::array<std::array<int, 5>, 5> datanew4{};
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    int temp = parent_status->board[j][i];
                    if (temp && temp < 7) {
                        datanew3[i][j] = temp + 6;
                    } else if (temp > 6) {
                        datanew4[i][j] = temp - 6;
                    }
                }
            }
            DQ.push_back(datanew4);
            DQ.push_back(datanew3);
            parent_status = parent_status->parent;
        } else {
            break;
        }
    }
}

torch::Tensor queueToTensor(std::deque<std::array<std::array<int, 5>, 5>> &DQ) {
    torch::Tensor tensor = torch::zeros({10, 5, 5}, torch::kFloat);
    int i = 0;
    for (const auto &board: DQ) {
        torch::Tensor tBoard = torch::from_blob((int *) board.data(), {5, 5}, torch::kInt32);
        tensor[i++] = tBoard.clone();
    }
    return tensor;
}

//返回最佳子节点
std::shared_ptr<Board> BestChild(std::shared_ptr<Board> v, double c, double k) {
    std::shared_ptr<Board> q;
    double max = -1.0e10;
    if (v->color == 0) {
        std::deque<std::array<std::array<int, 5>, 5>> Q;
        get_alldata_5blue(v, Q);
        torch::Tensor datanew = queueToTensor(Q);
        datanew = datanew.to(device);
        datanew = datanew.unsqueeze(0);
        torch::Tensor outnew = model.forward({datanew}).toTensor();
        torch::Tensor out = torch::zeros({25});
        out = out.to(device);
        outnew = outnew.squeeze();
        for (int i = 0; i < 6; ++i) {
            torch::Tensor oneStatus = outnew.slice(0, i * 4, (i + 1) * 4);
            out.slice(0, i * 4, (i + 1) * 4) = 3 * torch::softmax(oneStatus, 0);
            for (int index = i * 4; index < (i + 1) * 4; ++index) {
                if (index < 3) {
                    if (out[index].item<double>() < 0.2) {
                        out[index] = 0.2;
                    }
                }
            }
        }
        for (int i = 0; i < v->child.size(); i++) {
            // 根据v->child[i]->chess[0](棋子)和v->child[i]->chess[1](方向0 1 2)找到对应的概率
            double prob = out[(v->child[i]->chess[0] - 7) * 4 + v->child[i]->chess[1]].item<double>();
            double UCB = v->child[i]->quality / (double) v->child[i]->visit_times +
                         c * sqrt(2 * log((double) v->visit_times) / (double) v->child[i]->visit_times) +
                         k * prob / (double) v->child[i]->visit_times;
            if (UCB > max) {
                q = v->child[i];
                max = UCB;
            }
        }
    } else {
        std::deque<std::array<std::array<int, 5>, 5>> Q;
        get_alldata_5(v, Q);
        torch::Tensor datanew = queueToTensor(Q);
        datanew = datanew.to(device);
        datanew = datanew.unsqueeze(0);
        torch::Tensor outnew = model.forward({datanew}).toTensor();
        torch::Tensor out = torch::zeros({25});
        out = out.to(device);
        outnew = outnew.squeeze();
        for (int i = 0; i < 6; ++i) {
            torch::Tensor oneStatus = outnew.slice(0, i * 4, (i + 1) * 4);
            out.slice(0, i * 4, (i + 1) * 4) = 3 * torch::softmax(oneStatus, 0);
            for (int index = i * 4; index < (i + 1) * 4; ++index) {
                if (index < 3) {
                    if (out[index].item<double>() < 0.2) {
                        out[index] = 0.2;
                    }
                }
            }
        }
        for (int i = 0; i < v->child.size(); i++) {
            // 根据v->child[i]->chess[0](棋子)和v->child[i]->chess[1](方向0 1 2)找到对应的概率
            double prob = out[(v->child[i]->chess[0] - 1) * 4 + v->child[i]->chess[1]].item<double>();
            double UCB = v->child[i]->quality / (double) v->child[i]->visit_times +
                         c * sqrt(2 * log((double) v->visit_times) / (double) v->child[i]->visit_times) +
                         k * prob / (double) v->child[i]->visit_times;
            if (UCB > max) {
                q = v->child[i];
                max = UCB;
            }
        }
    }
    return q;
}

std::shared_ptr<Board> MostWin(std::shared_ptr<Board> v, double qualities[6][3]) {
    std::shared_ptr<Board> p;
    double max = -1.0;
    for (int i = 0; i < v->child.size(); i++) {
        qualities[i][0] = v->child[i]->chess[0];
        qualities[i][1] = v->child[i]->chess[1];
        qualities[i][2] = v->child[i]->quality;
        if (v->child[i]->quality - max > 0.0) {
            max = v->child[i]->quality;
            p = v->child[i];
        }
    }
    return p;
}

void oneMove(int color, int newboard[5][5], int chess, int direction) {
    int x, y;
    if (color == 0) {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (newboard[i][j] == chess) {
                    x = i;
                    y = j;
                    break;
                }
        if (direction == 0) {
            newboard[x][y + 1] = chess;
            newboard[x][y] = 0;
        } else if (direction == 1) {
            newboard[x + 1][y] = chess;
            newboard[x][y] = 0;
        } else {
            newboard[x + 1][y + 1] = chess;
            newboard[x][y] = 0;
        }
    } else {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (newboard[i][j] == chess + 6) {
                    x = i;
                    y = j;
                    break;
                }
        if (direction == 0) {
            newboard[x][y - 1] = chess + 6;
            newboard[x][y] = 0;
        } else if (direction == 1) {
            newboard[x - 1][y] = chess + 6;
            newboard[x][y] = 0;
        } else {
            newboard[x - 1][y - 1] = chess + 6;
            newboard[x][y] = 0;
        }
    }
}

//拓展
std::shared_ptr<Board> Expand(std::shared_ptr<Board> v) {
    std::vector<int> posChess[2];
    bool flag;
    for (int i = 0; i < v->posStep->size(); i++) {
        flag = false;
        for (int j = 0; j < v->child.size(); j++) {
            if (v->child[j]->chess[0] == v->posStep[0][i] && v->child[j]->chess[1] == v->posStep[1][i]) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            posChess[0].push_back(v->posStep[0][i]);
            posChess[1].push_back(v->posStep[1][i]);
        }
    }
    std::uniform_int_distribution<int> dist(0, posChess[0].size() - 1);
    int index = dist(generator);
    int newBoard[5][5];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            newBoard[i][j] = v->board[i][j];
    int pos[2];
    pos[0] = posChess[0][index];
    pos[1] = posChess[1][index];
    oneMove(v->color, newBoard, pos[0], pos[1]);
    std::shared_ptr<Board> newChild(new Board(v, !(bool) v->color, newBoard, pos));
    v->child.push_back(newChild);
    return newChild;
}

//搜索策略
//判断是否为终止节点
bool is_Terminal(std::shared_ptr<Board> v) {
    int blue, red;
    blue = red = 0;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) {
            if (v->board[i][j] >= 1 && v->board[i][j] <= 6)
                red++;
            else if (v->board[i][j] >= 7)
                blue++;
        }
    if (red == 0 || blue == 0 || v->board[0][0] >= 7 || (v->board[4][4] >= 1 && v->board[4][4] <= 6))
        return true;
    else
        return false;
}

bool is_Expanded(std::shared_ptr<Board> v) {
    if (v->posStep[0].size() > v->child.size())
        return false;
    else
        return true;
}

std::shared_ptr<Board> Treepolicy(std::shared_ptr<Board> v) {
    while (!is_Terminal(v)) {
        if (!is_Expanded(v))
            return Expand(v);
        else
            v = BestChild(v, coe, kilo);
    }
    return v;
}

int random_choices(const std::vector<double> &weights) {
    // 初始化随机数生成器
    std::random_device rdd;
    std::mt19937 gene(rdd());
    // 创建离散分布
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    // 根据权重选择元素
    return dist(gene);
}

template<typename T>
std::vector<T> SoftMax(const std::vector<T> &input, double temperature = 1.0) {
    std::vector<T> output(input.size());
    T maxElement = *std::max_element(input.begin(), input.end());
    T sum = 0;
    // 计算指数的和
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp((input[i] - maxElement) / temperature); // 添加温度参数
        sum += output[i];
    }
    // 归一化以得到概率分布
    for (T &val: output) {
        val /= sum;
    }
    return output;
}

bool RandChoice(std::shared_ptr<Board> v, bool col, std::mt19937 &gen) {
    int ch[7] = {0};
    int x, y;
    int DICE = dis(gen);
    if (col == 0) {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) {
                int temp = v->board[i][j];
                if (temp >= 1 && temp <= 6)
                    ch[temp]++;
            }
        if (ch[DICE] == 1) {
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    if (v->board[i][j] == DICE) {
                        x = i;
                        y = j;
                        break;
                    }
            if (x == 4) {
                v->board[x][y + 1] = DICE;
                v->board[x][y] = 0;
            } else if (y == 4) {
                v->board[x + 1][y] = DICE;
                v->board[x][y] = 0;
            } else {
                int direct;
                std::vector<double> max_problis;
                for (int index = 0; index < 3; index++) {
                    int newBoard[5][5];
                    for (int i = 0; i < 5; i++)
                        for (int j = 0; j < 5; j++)
                            newBoard[i][j] = v->board[i][j];
                    if (index == 0) {
                        newBoard[x][y + 1] = DICE;
                    } else if (index == 1) {
                        newBoard[x + 1][y] = DICE;
                    } else {
                        newBoard[x + 1][y + 1] = DICE;
                    }
                    newBoard[x][y] = 0;
                    FakeBoard FB(newBoard);
                    double EVS = FB.getScore(0, betaone, lamda);
                    max_problis.push_back(EVS);
                }
                direct = random_choices(SoftMax(max_problis, tempo));
                if (direct == 0) {
                    v->board[x][y + 1] = DICE;
                } else if (direct == 1) {
                    v->board[x + 1][y] = DICE;
                } else {
                    v->board[x + 1][y + 1] = DICE;
                }
                v->board[x][y] = 0;
            }
        } else {
            int validchess[2] = {-1, -1};
            std::vector<double> max_problis;
            std::vector<std::vector<int>> consider;
            for (int i = DICE - 1; i > 0; i--) {
                if (ch[i] == 1) {
                    validchess[0] = i;
                    break;
                }
            }
            for (int i = DICE + 1; i < 7; i++) {
                if (ch[i] == 1) {
                    validchess[1] = i;
                    break;
                }
            }
            for (int i = 0; i < 2; i++) {
                if (validchess[i] == -1)
                    continue;
                for (int j = 0; j < 5; j++)
                    for (int k = 0; k < 5; k++) {
                        if (v->board[j][k] == validchess[i]) {
                            x = j;
                            y = k;
                            break;
                        }
                    }
                if (x == 4) {
                    int newBoard[5][5];
                    for (int m = 0; m < 5; m++)
                        for (int n = 0; n < 5; n++)
                            newBoard[m][n] = v->board[m][n];
                    newBoard[x][y + 1] = validchess[i];
                    newBoard[x][y] = 0;
                    FakeBoard FB(newBoard);
                    double EVS = FB.getScore(0, betaone, lamda);
                    max_problis.push_back(EVS);
                    std::vector<int> pos = {x, y, 0, validchess[i]};
                    consider.push_back(pos);
                } else if (y == 4) {
                    int newBoard[5][5];
                    for (int m = 0; m < 5; m++)
                        for (int n = 0; n < 5; n++)
                            newBoard[m][n] = v->board[m][n];
                    newBoard[x + 1][y] = validchess[i];
                    newBoard[x][y] = 0;
                    FakeBoard FB(newBoard);
                    double EVS = FB.getScore(0, betaone, lamda);
                    max_problis.push_back(EVS);
                    std::vector<int> pos = {x, y, 1, validchess[i]};
                    consider.push_back(pos);
                } else {
                    for (int index = 0; index < 3; index++) {
                        int newBoard[5][5];
                        for (int m = 0; m < 5; m++)
                            for (int n = 0; n < 5; n++)
                                newBoard[m][n] = v->board[m][n];
                        if (index == 0) {
                            newBoard[x][y + 1] = DICE;
                        } else if (index == 1) {
                            newBoard[x + 1][y] = DICE;
                        } else {
                            newBoard[x + 1][y + 1] = DICE;
                        }
                        newBoard[x][y] = 0;
                        FakeBoard FB(newBoard);
                        double EVS = FB.getScore(0, betaone, lamda);
                        max_problis.push_back(EVS);
                        std::vector<int> pos = {x, y, index, validchess[i]};
                        consider.push_back(pos);
                    }
                }
            }
            int direct = random_choices(SoftMax(max_problis, tempo));
            std::vector<int> pos = consider[direct];
            if (pos[2] == 0) {
                v->board[pos[0]][pos[1] + 1] = pos[3];
            } else if (pos[2] == 1) {
                v->board[pos[0] + 1][pos[1]] = pos[3];
            } else {
                v->board[pos[0] + 1][pos[1] + 1] = pos[3];
            }
            v->board[pos[0]][pos[1]] = 0;
        }
    } else {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) {
                int temp = v->board[i][j];
                if (temp >= 7)
                    ch[temp - 6]++;
            }
        if (ch[DICE] == 1) {
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    if (v->board[i][j] == DICE + 6) {
                        x = i;
                        y = j;
                        break;
                    }
            if (x == 0) {
                v->board[x][y - 1] = DICE + 6;
                v->board[x][y] = 0;
            } else if (y == 0) {
                v->board[x - 1][y] = DICE + 6;
                v->board[x][y] = 0;
            } else {
                int direct;
                std::vector<double> max_problis;
                for (int index = 0; index < 3; index++) {
                    int newBoard[5][5];
                    for (int i = 0; i < 5; i++)
                        for (int j = 0; j < 5; j++)
                            newBoard[i][j] = v->board[i][j];
                    if (index == 0) {
                        newBoard[x][y - 1] = DICE + 6;
                    } else if (index == 1) {
                        newBoard[x - 1][y] = DICE + 6;
                    } else {
                        newBoard[x - 1][y - 1] = DICE + 6;
                    }
                    newBoard[x][y] = 0;
                    FakeBoard FB(newBoard);
                    double EVS = FB.getScore(1, betaone, lamda);
                    max_problis.push_back(EVS);
                }
                direct = random_choices(SoftMax(max_problis, tempo));
                if (direct == 0) {
                    v->board[x][y - 1] = DICE + 6;
                } else if (direct == 1) {
                    v->board[x - 1][y] = DICE + 6;
                } else {
                    v->board[x - 1][y - 1] = DICE + 6;
                }
                v->board[x][y] = 0;
            }
        } else {
            int validchess[2] = {-1, -1};
            std::vector<double> max_problis;
            std::vector<std::vector<int>> consider;
            for (int i = DICE - 1; i > 0; i--) {
                if (ch[i] == 1) {
                    validchess[0] = i + 6;
                    break;
                }
            }
            for (int i = DICE + 1; i < 7; i++) {
                if (ch[i] == 1) {
                    validchess[1] = i + 6;
                    break;
                }
            }
            for (int i = 0; i < 2; i++) {
                if (validchess[i] == -1)
                    continue;
                for (int j = 0; j < 5; j++)
                    for (int k = 0; k < 5; k++) {
                        if (v->board[j][k] == validchess[i]) {
                            x = j;
                            y = k;
                            break;
                        }
                    }
                if (x == 0) {
                    int newBoard[5][5];
                    for (int m = 0; m < 5; m++)
                        for (int n = 0; n < 5; n++)
                            newBoard[m][n] = v->board[m][n];
                    newBoard[x][y - 1] = validchess[i];
                    newBoard[x][y] = 0;
                    FakeBoard FB(newBoard);
                    double EVS = FB.getScore(1, betaone, lamda);
                    max_problis.push_back(EVS);
                    std::vector<int> pos = {x, y, 0, validchess[i]};
                    consider.push_back(pos);
                } else if (y == 0) {
                    int newBoard[5][5];
                    for (int m = 0; m < 5; m++)
                        for (int n = 0; n < 5; n++)
                            newBoard[m][n] = v->board[m][n];
                    newBoard[x - 1][y] = validchess[i];
                    newBoard[x][y] = 0;
                    FakeBoard FB(newBoard);
                    double EVS = FB.getScore(1, betaone, lamda);
                    max_problis.push_back(EVS);
                    std::vector<int> pos = {x, y, 1, validchess[i]};
                    consider.push_back(pos);
                } else {
                    for (int index = 0; index < 3; index++) {
                        int newBoard[5][5];
                        for (int m = 0; m < 5; m++)
                            for (int n = 0; n < 5; n++)
                                newBoard[m][n] = v->board[m][n];
                        if (index == 0) {
                            newBoard[x][y - 1] = DICE + 6;
                        } else if (index == 1) {
                            newBoard[x - 1][y] = DICE + 6;
                        } else {
                            newBoard[x - 1][y - 1] = DICE + 6;
                        }
                        newBoard[x][y] = 0;
                        FakeBoard FB(newBoard);
                        double EVS = FB.getScore(1, betaone, lamda);
                        max_problis.push_back(EVS);
                        std::vector<int> pos = {x, y, index, validchess[i]};
                        consider.push_back(pos);
                    }
                }
            }
            int direct = random_choices(SoftMax(max_problis, tempo));
            std::vector<int> pos = consider[direct];
            if (pos[2] == 0) {
                v->board[pos[0]][pos[1] - 1] = pos[3];
            } else if (pos[2] == 1) {
                v->board[pos[0] - 1][pos[1]] = pos[3];
            } else {
                v->board[pos[0] - 1][pos[1] - 1] = pos[3];
            }
            v->board[pos[0]][pos[1]] = 0;
        }
    }
    return !col;
}

int simulate(std::shared_ptr<Board> v, std::mt19937 &gen) {
    int origin[5][5];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            origin[i][j] = v->board[i][j];
    bool tempColor = v->color;
    while (!is_Terminal(v)) {
        tempColor = RandChoice(v, tempColor, gen);
    }
    //返回模拟结果
    int result;
    int blue, red;
    blue = red = 0;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) {
            if (v->board[i][j] >= 1 && v->board[i][j] <= 6)
                red++;
            else if (v->board[i][j] >= 7)
                blue++;
        }
    if (blue == 0 || (v->board[4][4] >= 1 && v->board[4][4] <= 6))
        result = 0;
    else
        result = 1;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            v->board[i][j] = origin[i][j];
    return result;
}

void Backup(std::shared_ptr<Board> v, int result) {
    while (v) {
        v->visit_times++;                            //到底是谁的visit次数？
        if ((v->color ^ result))
            v->win_time++;
        v->quality = double(v->win_time) / double(v->visit_times);        //到底是谁的visit次数？
        v = v->parent;
    }
}

void DeleteAll(std::shared_ptr<Board> v) {
    for (int i = 0; i < v->child.size(); i++) {
        DeleteAll(v->child[i]);
    }
    std::vector<std::shared_ptr<Board>>().swap(v->child);
}

std::shared_ptr<Board> UCTP(int dice, int board[5][5], int &iter, double qualities[6][3]) {
    std::shared_ptr<Board> root;
    std::shared_ptr<Board> none;
    if (dice <= 6) {
        root = std::make_shared<Board>(none, 0, board, dice);
    } else {
        root = std::make_shared<Board>(none, 1, board, dice);
    }

    int iteration = 0;
    iter = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            qualities[i][j] = -1;
        }
    }

    int Move_Num = root->posStep[0].size();
    omp_set_num_threads(Move_Num);

#pragma omp parallel for reduction(+:iteration) default(none) shared(Move_Num, device, root)
    for (int index = 0; index < Move_Num; index++) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + omp_get_thread_num();
        std::mt19937 gen(seed);
        int newBoard[5][5];
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                newBoard[i][j] = root->board[i][j];
        int pos[2];
        pos[0] = root->posStep[0][index];
        pos[1] = root->posStep[1][index];
        oneMove(root->color, newBoard, pos[0], pos[1]);
        std::shared_ptr<Board> newChild;
        newChild = std::make_shared<Board>(root, !(bool) root->color, newBoard, pos);
#pragma omp critical
        {
            root->child.push_back(newChild);
        }

        auto start = std::chrono::steady_clock::now();
        auto end = std::chrono::steady_clock::now();
        auto diff_time = std::chrono::duration<double, std::milli>(end - start).count();
        while (diff_time < 100000) {
            std::shared_ptr<Board> p;
            p = Treepolicy(newChild);
            int result = simulate(p, gen);
            Backup(p, result);
            iteration++;
            end = std::chrono::steady_clock::now();
            diff_time = std::chrono::duration<double, std::milli>(end - start).count();
        }
    }
    iter = iteration;
    std::shared_ptr<Board> best;
    best = MostWin(root, qualities);
    return best;
}


void to_int(std::string s, int begin, int end, int *a, int line) {
    int i = begin;
    int j = begin + 1;
    int t = 0;
    while (j < end + 1) {
        if (s[j] == ',' || s[j] == ']') {
            std::string x = s.substr(i, j - i);
            a[line * 5 + t] = atoi(x.c_str());
            t++;
            i = j + 2;
        }
        j++;
    }
}

CEinstein::CEinstein() {
    this->clientsocket.connectServer();
    clientsocket.sendMsg(ID);
}

CEinstein::~CEinstein() {
    this->clientsocket.close();
}

int CEinstein::parse(std::string s) {
    int k = s.find("], [", 0);
    int l = s.find("], [", k + 4);
    int m = s.find("], [", l + 4);
    int n = s.find("], [", m + 4);
    int p = s.find("]]|", n + 4);
    to_int(s, 2, k, chessboard, 0);
    to_int(s, k + 4, l, chessboard, 1);
    to_int(s, l + 4, m, chessboard, 2);
    to_int(s, m + 4, n, chessboard, 3);
    to_int(s, n + 4, p, chessboard, 4);
    std::string d = s.substr(p + 3, s.length() - p - 3);
    dice = atoi(d.c_str());
    if (logger.size() == 0) {
        begin_time = clock();
        return 0;
    } else {
        std::string g = logger.back();
        int last[25];
        int f = g.find("[[", 0);
        int k = g.find("], [", f + 2);
        int l = g.find("], [", k + 4);
        int m = g.find("], [", l + 4);
        int n = g.find("], [", m + 4);
        int p = g.find("]]|", n + 4);
        to_int(g, f + 2, k, last, 0);
        to_int(g, k + 4, l, last, 1);
        to_int(g, l + 4, m, last, 2);
        to_int(g, m + 4, n, last, 3);
        to_int(g, n + 4, p, last, 4);
        int diff = 0;
        for (int i = 0; i < 25; i++)
            if (chessboard[i] != last[i])
                diff++;
        if (diff <= 4)
            return 0;
        else {
            int ze = g.rfind(" ");
            int pr = g.rfind("|");
            int chess = std::stoi(g.substr(ze + 1, pr - ze - 1));
            std::string operation = g.substr(pr + 1, g.length() - pr - 1);
            int t = 0;
            while (last[t] != chess)
                t++;
            if (operation == "up") {
                last[t - 5] = chess;
                last[t] = 0;
            } else if (operation == "left") {
                last[t - 1] = chess;
                last[t] = 0;
            } else if (operation == "leftup") {
                last[t - 6] = chess;
                last[t] = 0;
            } else if (operation == "down") {
                last[t + 5] = chess;
                last[t] = 0;
            } else if (operation == "right") {
                last[t + 1] = chess;
                last[t] = 0;
            } else {
                last[t + 6] = chess;
                last[t] = 0;
            }
            end_time = clock();
            double lasting = difftime(end_time, begin_time) / (double) CLOCKS_PER_SEC;
            begin_time = clock();
            if (chess >= 7) {
                int red = 0;
                for (int i = 0; i < 25; i++)
                    if (last[i] >= 1 && last[i] <= 6)
                        red++;
                if (red == 0 || last[0] == chess) {
                    logging("Game over! Blue wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                } else {
                    logging("Game over! Red wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                }
            } else {
                int blue = 0;
                for (int i = 0; i < 25; i++)
                    if (last[i] >= 7 && last[i] <= 12)
                        blue++;
                if (blue == 0 || last[24] == chess) {
                    logging("Game over! Red wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                } else {
                    logging("Game over! Blue wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                }
            }
        }
    }
}

int CEinstein::logging(std::string s) {
    logger.push_back(s);
    std::cout << s << std::endl;
    return 0;
}

int CEinstein::writelog() {
    std::ofstream fout;
    filename = std::to_string(pt->tm_year + 1900) + "-" + std::to_string(pt->tm_mon + 1) + "-" +
               std::to_string(pt->tm_mday) + "-181860013.log";
    fout.open(filename);
    std::list<std::string>::iterator itor;
    itor = logger.begin();
    while (itor != logger.end())
        fout << *itor++ << std::endl;
    return 0;
}

int CEinstein::handle() {
    int t = clientsocket.recvMsg();
    if (t == 0) {
        char *a = clientsocket.getRecvMsg();
        std::string rec = a;
        parse(rec);
        int board[5][5];
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                board[i][j] = chessboard[i * 5 + j];
        //std::string str = fool(dice, board);
        std::shared_ptr<Board> best;
        int iter;
        double qualities[6][3];
        best = UCTP(dice, board, iter, qualities);
        std::string str;
        if (dice <= 6) {
            if (best->chess[1] == 0)
                str = std::to_string(best->chess[0]) + "|right";
            else if (best->chess[1] == 1)
                str = std::to_string(best->chess[0]) + "|down";
            else if (best->chess[1] == 2)
                str = std::to_string(best->chess[0]) + "|rightdown";
        } else {
            if (best->chess[1] == 0)
                str = std::to_string(best->chess[0] + 6) + "|left";
            else if (best->chess[1] == 1)
                str = std::to_string(best->chess[0] + 6) + "|up";
            else if (best->chess[1] == 2)
                str = std::to_string(best->chess[0] + 6) + "|leftup";
        }
        now_time = time(NULL);
        clientsocket.sendMsg(str.c_str());
        std::string m = rec + " operation " + str;
        pt = localtime(&now_time);
        std::string log = std::to_string(pt->tm_year + 1900) + "-" + std::to_string(pt->tm_mon + 1) + "-" +
                          std::to_string(pt->tm_mday) + " " + std::to_string(pt->tm_hour) + "-" +
                          std::to_string(pt->tm_min) + "-" + std::to_string(pt->tm_sec) + " : " + m;
        logging(log);
        std::cout << best->quality * 100.0 << "%" << iter << std::endl << std::endl;
        DeleteAll(best->parent);
        return 1;
    } else {
        std::string g = logger.back();
        int last[25];
        int f = g.find("[[", 0);
        int k = g.find("], [", f + 2);
        int l = g.find("], [", k + 4);
        int m = g.find("], [", l + 4);
        int n = g.find("], [", m + 4);
        int p = g.find("]]|", n + 4);
        to_int(g, f + 2, k, last, 0);
        to_int(g, k + 4, l, last, 1);
        to_int(g, l + 4, m, last, 2);
        to_int(g, m + 4, n, last, 3);
        to_int(g, n + 4, p, last, 4);
        int diff = 0;
        for (int i = 0; i < 25; i++)
            if (chessboard[i] != last[i])
                diff++;
        if (diff <= 4)
            return 0;
        else {
            int ze = g.rfind(" ");
            int pr = g.rfind("|");
            int chess = std::stoi(g.substr(ze + 1, pr - ze - 1));
            std::string operation = g.substr(pr + 1, g.length() - pr - 1);
            int t = 0;
            while (last[t] != chess)
                t++;
            if (operation == "up") {
                last[t - 5] = chess;
                last[t] = 0;
            } else if (operation == "left") {
                last[t - 1] = chess;
                last[t] = 0;
            } else if (operation == "leftup") {
                last[t - 6] = chess;
                last[t] = 0;
            } else if (operation == "down") {
                last[t + 5] = chess;
                last[t] = 0;
            } else if (operation == "right") {
                last[t + 1] = chess;
                last[t] = 0;
            } else {
                last[t + 6] = chess;
                last[t] = 0;
            }
            end_time = clock();
            double lasting = difftime(end_time, begin_time) / (double) CLOCKS_PER_SEC;
            if (chess >= 7) {
                int red = 0;
                for (int i = 0; i < 25; i++)
                    if (last[i] >= 1 && last[i] <= 6)
                        red++;
                if (red == 0 || last[0] == chess) {
                    logging("Game over! Blue wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                } else {
                    logging("Game over! Red wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                }
            } else {
                int blue = 0;
                for (int i = 0; i < 25; i++)
                    if (last[i] >= 7 && last[i] <= 12)
                        blue++;
                if (blue == 0 || last[24] == chess) {
                    logging("Game over! Red wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                } else {
                    logging("Game over! Blue wins! Time: " + std::to_string(lasting) + "s");
                    return 0;
                }
            }
        }
        return 0;
    }
}
