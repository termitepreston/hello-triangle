#pragma once

#include "window.h"

#include <cassert>
#include <iostream>

void WindowDestroyer::operator()(GLFWwindow* window) const
{
    assert(window != nullptr);

    std::cout << "[info] destroying window." << std::endl;
    glfwDestroyWindow(window);
    window = nullptr;
    std::cout << "[info] terminating glfw." << std::endl;
    glfwTerminate();
}

AppWindow::AppWindow()
    : AppWindow(title, size)
{
}

AppWindow::AppWindow(std::string_view windowTitle, Size windowSize)
    : title { windowTitle }
    , size { windowSize }
{
    std::cout << "[info] initializing glfw." << std::endl;
    glfwInit();

    // tell glfw that we are not interested in opengl.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // no need for unnecessary code to handle window resizes (for now).
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    std::cout << "[info] creating window." << std::endl;
    window.reset(glfwCreateWindow(size.width, size.height, title.c_str(), nullptr, nullptr));


    if (window == nullptr) {
        throw std::runtime_error("[error] failed constructing window object.");
    }
}

GLFWwindow* AppWindow::getWindow() const
{
    return window.get();
}