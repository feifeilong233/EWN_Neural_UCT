cmake_minimum_required(VERSION 3.26)
project(EWN_Neural)

set(CMAKE_CXX_STANDARD 14)

set(Torch_DIR "C:/libtorch/share/cmake/Torch")

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(EWN_Neural
        ClientSocket.cpp
        ClientSocket.h
        Define.h
        CFakeBoard.h
        CEinstein.cpp
        CEinstein.h
        main.cpp)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        OpenMP::OpenMP_CXX
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
