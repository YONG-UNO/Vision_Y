cmake_minimum_required(VERSION 4.2)
project(Vision_Y)

set(CMAKE_CXX_STANDARD 20)

# 自动查找OpenCV
find_package(OpenCV REQUIRED)

# 头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 生成可执行文件
add_executable(Vision_Y
        OpticFlow.cpp
#        main.cpp
)

# 连接的哦阿OpenCV库
target_link_libraries(Vision_Y ${OpenCV_LIBS})
