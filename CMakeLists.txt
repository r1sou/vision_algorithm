cmake_minimum_required(VERSION 3.20)
project(vision_algorithm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-pthread -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# set(ARCH "aarch")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("Detect arch -> aarch64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
	message("Detect arch -> x86")
else()
    message("Unknow arch")
endif()

find_package(OpenCV REQUIRED)
find_package(PCL 1.12 REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/fmt/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/ThreadPool/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/nlohmann/include
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/dnn/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/dnn/lib)

add_definitions(-DCAMERA="${CMAKE_CURRENT_SOURCE_DIR}/assets/config/camera.json")
add_definitions(-DMODEL="${CMAKE_CURRENT_SOURCE_DIR}/assets/config/model.json")

function(add_vision_executable target_name)
    add_executable(${target_name} ${ARGN})
    ament_target_dependencies(${target_name}
        rclcpp
        cv_bridge
        std_msgs
        sensor_msgs
        message_filters
        geometry_msgs
    )
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        target_link_libraries(${target_name}
            ${PCL_LIBRARIES}
            ${OpenCV_LIBRARIES}
            pthread
            dnn
        )
    else()
        target_link_libraries(${target_name}
            ${PCL_LIBRARIES}
            ${OpenCV_LIBRARIES}
            pthread
        )
    endif()

    install(TARGETS ${target_name}
        DESTINATION lib/${PROJECT_NAME}
    )
endfunction()

add_vision_executable(vision
    src/utils/file.cpp
    src/utils/imgproc.cpp
    src/core/node.cpp
    src/main.cpp
)

add_vision_executable(test_camera
    src/utils/file.cpp
    src/utils/imgproc.cpp
    src/core/node.cpp
    src/test/test_camera.cpp
)

add_vision_executable(test_model
    src/utils/file.cpp
    src/utils/imgproc.cpp
    src/core/node.cpp
    src/core/model.cpp
    src/test/test_model.cpp
)

ament_package()