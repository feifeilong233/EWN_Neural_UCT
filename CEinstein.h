#ifndef EWN_BRAVO_UCTH_H
#define EWN_BRAVO_UCTH_H

#include<vector>
#include<cmath>
#include<memory>
#include <iostream>
#include <ctime>
#include <chrono>
#include <tuple>
#include <omp.h>
#include <random>
#include "CFakeBoard.h"
#include <torch/torch.h>
#include <torch/script.h>

#include<fstream>
#include <list>
#include "ClientSocket.h"


class CEinstein {
private:
    ClientSocket clientsocket;
    std::list<std::string> logger;
    std::string filename; // log filename
    int chessboard[25];
    int dice;

public:
    CEinstein();

    ~CEinstein();

    int parse(std::string s);

    int handle();

    int logging(std::string s);

    int writelog();
};

#endif //EWN_BRAVO_UCTH_H
