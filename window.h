#pragma once

#pragma once

#include <string>
#include <memory>

#include <GLFW/glfw3.h>

/*
        window object that encapsulates RAII concepts.
*/

struct Size {
    int width;
    int height;
};

struct WindowDestroyer {
    void operator()(GLFWwindow* window) const;
};

// according to isocpp guidelines a class should enforce an invariant.
// for this class the invariant is GLFWWindow* object.

class AppWindow {
public:
    AppWindow();
    AppWindow(std::string_view windowTitle, Size windowSize);

    GLFWwindow* getWindow() const;

    ~AppWindow() = default;

private:
    std::string title { "Default Window" };
    std::unique_ptr<GLFWwindow, WindowDestroyer> window { nullptr };
    Size size { 640, 480 };
};