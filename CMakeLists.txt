cmake_minimum_required(VERSION 3.21)
project(FinalNNCpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)


find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        REQUIRED)

add_executable(FinalNNCpp main.cpp Node.cpp Node.h Layer.cpp Layer.h Network.cpp Network.h LayerClasses/Dense.cpp LayerClasses/Dense.h)
target_link_libraries(FinalNNCpp
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        )