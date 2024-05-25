cmake_minimum_required(VERSION 3.16)
project(EWN_Neural)

set(CMAKE_CXX_STANDARD 14)

if (APPLE)
    set(OPENMP_LIBRARIES "/usr/local/Cellar/libomp/18.1.1/lib")
    set(OPENMP_INCLUDES "/usr/local/Cellar/libomp/18.1.1/include")
    if (CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif ()
endif ()

if (NOT APPLE)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(Torch_DIR "/home/wfl/libtorch/share/cmake/Torch")
endif ()

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

if (APPLE)
    include_directories("${OPENMP_INCLUDES}")
    link_directories("${OPENMP_LIBRARIES}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_executable(EWN_Neural
        ClientSocket.cpp
        ClientSocket.h
        Define.h
        Board.h
        CFakeBoard.h
        CEinstein.cpp
        CEinstein.h
        main.cpp)

if (NOT APPLE)
    target_link_libraries(${PROJECT_NAME}
            PUBLIC
            OpenMP::OpenMP_CXX
    )
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
