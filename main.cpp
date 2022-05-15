#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "window.h"

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    } else {
        std::cout << "[warn] cannot get pointer to function." << std::endl;
    }
}

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidation = false;
#else
const bool enableValidation = true;
#endif

enum class ShaderType { Vertex,
    Fragment };

static std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    ShaderType shaderType;
    std::string shaderTypeStr;

    if (filename.ends_with("vert.spv")) {
        shaderType = ShaderType::Vertex;
        shaderTypeStr = "vertex";
    } else {
        shaderType = ShaderType::Fragment;
        shaderTypeStr = "fragment";
    }

    if (!file.is_open()) {
        throw std::runtime_error("[error] cannot open file " + filename);
    }

    auto fileSize = file.tellg();
    std::cout << "[info] "
              << shaderTypeStr << " shader"
              << " file is " << static_cast<float>(fileSize) / 1024.0f << " kilobytes." << std::endl;
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> modes;
};

VkSurfaceFormatKHR chooseSwapChainFormat(const std::vector<VkSurfaceFormatKHR>& avialableFormats)
{
    for (const auto& availableFormat : avialableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }

        return avialableFormats[0];
    }
}

VkPresentModeKHR chooseSwapChainPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

class HelloTriangleApplication {
public:
    HelloTriangleApplication()
        : window { "Vulkan App", { .width = 640, .height = 480 } }
        , instance {}
        , debugMessenger {}
    {
    }
    void run()
    {
        initVulkan();
        mainLoop();
        cleanUp();
    }

private:
    AppWindow window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkSurfaceKHR surface;
    VkQueue presentQueue;
    VkQueue graphicsQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages {};
    std::vector<VkImageView> swapChainImageViews {};
    std::vector<VkFramebuffer> swapChainFramebuffers {};
    VkFormat swapChainFormat;
    VkExtent2D swapChainExtent;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkCommandPool commandPool; // write up on implicit destruction.
    VkCommandBuffer commandBuffer;
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;

private:
    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffer();
        createSyncObjects();
    }

    void createSyncObjects()
    {
        VkSemaphoreCreateInfo semaphoreCreateInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };

        VkFenceCreateInfo fenceCreateInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };

        if (vkCreateSemaphore(this->device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to create imageAvailableSemaphore.");
        }

        if (vkCreateSemaphore(this->device, &semaphoreCreateInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to create renderFinishedSemaphore.");
        }

        if (vkCreateFence(this->device, &fenceCreateInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to create inFlightFence.");
        }
    }

    void createCommandBuffer()
    {
        VkCommandBufferAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = this->commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

        if (vkAllocateCommandBuffers(this->device, &allocInfo, &this->commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to allocate command buffer.");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = nullptr
        };

        if (vkBeginCommandBuffer(this->commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to begin recording command buffer.");
        }

        /*
            This part is very important as it ties everything so far.
        */
        VkClearValue clearColor { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
        VkRenderPassBeginInfo renderPassBeginInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = this->renderPass,
            .framebuffer = swapChainFramebuffers[imageIndex],
            .renderArea = {
                .offset = { 0, 0 },
                .extent = this->swapChainExtent },
            .clearValueCount = 1,
            .pClearValues = &clearColor
        };

        vkCmdBeginRenderPass(this->commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(this->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->graphicsPipeline);

        vkCmdDraw(this->commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(this->commandBuffer);

        if (vkEndCommandBuffer(this->commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to end command recording process.");
        }
    }

    void createCommandPool()
    {
        auto queueFamilyIndices = findQueueFamilies(this->physicalDevice);
        VkCommandPoolCreateInfo commandPoolCreateInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
        };

        if (vkCreateCommandPool(this->device, &commandPoolCreateInfo, nullptr, &this->commandPool) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to create command pool");
        }
    }

    void createFramebuffers()
    {
        this->swapChainFramebuffers.resize(this->swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferCreateInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = this->renderPass,
                .attachmentCount = 1,
                .pAttachments = attachments,
                .width = swapChainExtent.width,
                .height = swapChainExtent.height,
                .layers = 1,
            };

            if (vkCreateFramebuffer(this->device, &framebufferCreateInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("[error] failed creating frame buffer object at index " + i);
            }
        }
    }

    void createRenderPass()
    {
        // needs clarification.
        VkAttachmentDescription colorAttachment {
            .format = this->swapChainFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        };

        VkAttachmentReference colorAttachmentRef {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        };

        VkSubpassDescription subpass {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef
        };

        VkRenderPassCreateInfo renderPassCreateInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass
        };

        if (vkCreateRenderPass(this->device, &renderPassCreateInfo, nullptr, &this->renderPass) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed creating vk render pass.");
        }
    }

    VkShaderModule createShaderModule(const std::vector<char>& byteCode)
    {
        VkShaderModuleCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = byteCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(byteCode.data())
        };

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(this->device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed creating shader module.");
        }

        return shaderModule;
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("assets/shaders/compiled/vert.spv");
        auto fragShaderCode = readFile("assets/shaders/compiled/frag.spv");

        auto vertShaderModule = createShaderModule(vertShaderCode);
        auto fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main"
        };

        VkPipelineShaderStageCreateInfo fragShaderCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main"
        };

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderCreateInfo, fragShaderCreateInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };

        VkViewport viewport {
            .x = 0.0f,
            .y = 0.0f,
            .width = (float)this->swapChainExtent.width,
            .height = (float)this->swapChainExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        };

        VkRect2D scissor {
            .offset = { 0, 0 },
            .extent = this->swapChainExtent
        };

        VkPipelineViewportStateCreateInfo viewportStateCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor
        };

        VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f,

        };

        VkPipelineMultisampleStateCreateInfo multiSampleStateCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE
        };

        VkPipelineColorBlendAttachmentState colorBlendAttachmentStateCreateInfo {
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachmentStateCreateInfo,
        };

        colorBlendStateCreateInfo.blendConstants[0] = 0.0f;
        colorBlendStateCreateInfo.blendConstants[1] = 0.0f;
        colorBlendStateCreateInfo.blendConstants[2] = 0.0f;
        colorBlendStateCreateInfo.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_LINE_WIDTH
        };

        VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr
        };

        if (vkCreatePipelineLayout(this->device, &pipelineLayoutCreateInfo, nullptr, &this->pipelineLayout) != VK_FALSE) {
            throw std::runtime_error("[error] can not create pipeline layout object.");
        }

        VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputCreateInfo,
            .pInputAssemblyState = &inputAssemblyCreateInfo,
            .pViewportState = &viewportStateCreateInfo,
            .pRasterizationState = &rasterizationStateCreateInfo,
            .pMultisampleState = &multiSampleStateCreateInfo,
            .pDepthStencilState = nullptr,
            .pColorBlendState = &colorBlendStateCreateInfo,
            .pDynamicState = nullptr,
            .layout = this->pipelineLayout,
            .renderPass = this->renderPass,
            .subpass = 0, // index of subpass.
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1
        };

        if (vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &this->graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to create graphics pipeline.");
        }

        vkDestroyShaderModule(this->device, vertShaderModule, nullptr);
        vkDestroyShaderModule(this->device, fragShaderModule, nullptr);
    }

    void createImageViews()
    {
        this->swapChainImageViews.resize(swapChainImages.size());
        auto i = 0;
        for (const auto& swapChainImage : this->swapChainImages) {
            VkImageViewCreateInfo createInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapChainImage,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = this->swapChainFormat,
                .components = {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY },
                .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }
            };

            if (vkCreateImageView(this->device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("[error] failed creating imageview number " + i);
            }
            i++;
        }
    }

    void createSwapChain()
    {
        auto swapChainDetails = querySwapChainSupport(this->physicalDevice);
        auto swapChainFormat = chooseSwapChainFormat(swapChainDetails.formats);
        auto swapChainPresentMode = chooseSwapChainPresentMode(swapChainDetails.modes);
        auto extent = chooseSwapExtent(swapChainDetails.capabilities);

        uint32_t imageCount = swapChainDetails.capabilities.minImageCount + 1;

        if (swapChainDetails.capabilities.maxImageCount > 0 && imageCount > swapChainDetails.capabilities.maxImageCount) {
            imageCount = swapChainDetails.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = this->surface,
            .minImageCount = imageCount,
            .imageFormat = swapChainFormat.format,
            .imageColorSpace = swapChainFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = swapChainDetails.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = swapChainPresentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE
        };

        auto indices = findQueueFamilies(this->physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily.value() != indices.presentFamily.value()) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        if (vkCreateSwapchainKHR(this->device, &createInfo, nullptr, &this->swapChain) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed creating swap chain.");
        }

        vkGetSwapchainImagesKHR(this->device, this->swapChain, &imageCount, nullptr);
        if (imageCount != 0) {
            swapChainImages.resize(imageCount);
            vkGetSwapchainImagesKHR(this->device, this->swapChain, &imageCount, this->swapChainImages.data());
        }

        this->swapChainFormat = swapChainFormat.format;
        this->swapChainExtent = swapChainExtent;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;

            glfwGetFramebufferSize(window.getWindow(), &width, &height);

            VkExtent2D actualExtent = {
                .width = static_cast<uint32_t>(width),
                .height = static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(this->instance, window.getWindow(), nullptr, &this->surface) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to create window surface.");
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice phyDevice)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phyDevice, this->surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(phyDevice, this->surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(phyDevice, this->surface, &formatCount, details.formats.data());
        }

        uint32_t presentCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(phyDevice, this->surface, &presentCount, nullptr);

        if (presentCount != 0) {
            details.modes.resize(presentCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(phyDevice, this->surface, &presentCount, details.modes.data());
        }

        return details;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice phyDevice)
    {
        uint32_t availableExtensionsCount = 0;
        vkEnumerateDeviceExtensionProperties(phyDevice, nullptr, &availableExtensionsCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(availableExtensionsCount);
        vkEnumerateDeviceExtensionProperties(phyDevice, nullptr, &availableExtensionsCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            std::cout << "[info] found extension with name : " << extension.extensionName << std::endl;
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    void createLogicalDevice()
    {
        auto indices = findQueueFamilies(this->physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),
            indices.presentFamily.value() };

        float queuePriorities = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &queuePriorities,
            };

            queueCreateInfos.push_back(queueCreateInfo);
        }

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = indices.graphicsFamily.value(),
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };

        VkPhysicalDeviceFeatures deviceFeatures {
            .robustBufferAccess = VK_FALSE
        };

        VkDeviceCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &deviceFeatures,
        };

        if (enableValidation) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        std::cout << "[info] just before (logical) device creation." << std::endl;

        if (vkCreateDevice(this->physicalDevice, &createInfo, nullptr, &this->device) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed creating logical device.");
        }

        vkGetDeviceQueue(this->device, indices.graphicsFamily.value(), 0, &this->graphicsQueue);
        vkGetDeviceQueue(this->device, indices.presentFamily.value(), 0, &this->presentQueue);
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;

        vkEnumeratePhysicalDevices(this->instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("[error] can not find vulkan capable device.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        // fill in available devices.
        vkEnumeratePhysicalDevices(this->instance, &deviceCount, devices.data());

        for (const auto& phyDevice : devices) {
            std::cout << "[info] found device : " << phyDevice << std::endl;
            if (isDeviceSuitable(phyDevice)) {
                this->physicalDevice = phyDevice;
            }
        }

        if (this->physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("[error] cannot find a suitable device.");
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice phyDevice)
    {
        std::cout << "[info] received phyDevice = " << phyDevice << " in isDeviceSuitable(...)" << std::endl;
        // query properties and features.
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(phyDevice, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(phyDevice, &deviceFeatures);

        QueueFamilyIndices indices = findQueueFamilies(phyDevice);
        auto extensionsSupported = checkDeviceExtensionSupport(phyDevice);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            auto swapChainSupported = querySwapChainSupport(phyDevice);
            swapChainAdequate = !swapChainSupported.modes.empty() && !swapChainSupported.formats.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice phyDevice)
    {
        std::cout << "[info] received phyDevice = " << phyDevice << " in findQueueFamilies(...)" << std::endl;

        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phyDevice, &queueFamilyCount, nullptr);

        std::cout << "[info] queuefamily count = " << queueFamilyCount << std::endl;

        std::vector<VkQueueFamilyProperties> queuefamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(phyDevice, &queueFamilyCount, queuefamilies.data());

        int i = 0;
        for (const auto& queueFamily : queuefamilies) {
            // find for queue family that supports presentation.
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(phyDevice, i, this->surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }
            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
            .pfnUserCallback = this->debugCallback,
            .pUserData = nullptr
        };
    }

    void setupDebugMessenger()
    {
        if (!enableValidation)
            return;
        VkDebugUtilsMessengerCreateInfoEXT createInfo {};
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(this->instance, &createInfo, nullptr, &this->debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("[error] cannot create debug messenger.");
        }
    }

    void drawFrame()
    {
        // outline of a frame.
        // vulkan users explictly orchestrate drawing, allocation,
        // and presentation on the gpu + cpu and gpu operations.

        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        vkResetCommandBuffer(commandBuffer, 0);
        recordCommandBuffer(commandBuffer, imageIndex);

        // submitting the command buffer.

        VkSemaphore waitSemaphores[] { imageAvailableSemaphore };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkCommandBuffer commandBuffers[] = { commandBuffer };
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };

        VkSubmitInfo submitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = commandBuffers,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = signalSemaphores
        };

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("[error] command buffer sumission to the graphics queue failed.");
        }
        
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window.getWindow())) {
            glfwPollEvents();
            drawFrame();
        }
    }

    void cleanUp()
    {
        std::cout << "[info] destroying image available semaphore." << std::endl;
        vkDestroySemaphore(this->device, this->imageAvailableSemaphore, nullptr);
        std::cout << "[info] destroying rendering finished semaphore." << std::endl;
        vkDestroySemaphore(this->device, this->renderFinishedSemaphore, nullptr);
        std::cout << "[info] destroying inflight frame fence." << std::endl;
        vkDestroyFence(this->device, this->inFlightFence, nullptr);
        std::cout << "[info] destroying command pool." << std::endl;
        vkDestroyCommandPool(this->device, this->commandPool, nullptr);
        std::cout << "[info] destroying framebuffers" << std::endl;
        size_t i = 0;
        for (auto framebuffer : this->swapChainFramebuffers) {
            vkDestroyFramebuffer(this->device, framebuffer, nullptr);
            std::cout << "\t"
                      << "destroyed framebuffer object at index "
                      << i << std::endl;
            i++;
        }
        std::cout << "[info] destroying graphics pipeline (VkPipeline) object." << std::endl;
        vkDestroyPipeline(this->device, this->graphicsPipeline, nullptr);
        std::cout << "[info] destroying pipeline layout object." << std::endl;
        vkDestroyPipelineLayout(this->device, this->pipelineLayout, nullptr);
        std::cout << "[info] destroying render pass object." << std::endl;
        vkDestroyRenderPass(this->device, this->renderPass, nullptr);
        std::cout << "[info] destroying image views..." << std::endl;
        size_t j = 0;
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(this->device, imageView, nullptr);
            std::cout << "\t"
                      << "destroyed image view number " << j << std::endl;
            j++;
        }
        if (enableValidation) {
            DestroyDebugUtilsMessengerEXT(this->instance, this->debugMessenger, nullptr);
        }
        std::cout << "[info] destroying vk swap chain." << std::endl;
        vkDestroySwapchainKHR(this->device, this->swapChain, nullptr);
        std::cout << "[info] destroying vk device." << std::endl;
        vkDestroyDevice(this->device, nullptr);
        std::cout << "[info] destroying vk window surface." << std::endl;
        vkDestroySurfaceKHR(this->instance, this->surface, nullptr);
        std::cout << "[info] destroying vk instance." << std::endl;
        vkDestroyInstance(this->instance, nullptr);
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    void showExtensions()
    {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);

        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::cout << "[info] available extensions" << std::endl;
        for (const auto& extension : extensions) {
            std::cout << "\t" << extension.extensionName << std::endl;
        }
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::cout << "[info] glfwExtensions = " << glfwExtensions << std::endl;
        std::cout << "[info] glfwExtensionCount = " << glfwExtensionCount << std::endl;
        std::cout << "[info] glfwExtensionCount + glfwExtensions = " << glfwExtensionCount + glfwExtensions << std::endl;

        // this code needs some attention.
        // may be start pointer...end pointer.
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        std::cout << "[info] len(extensions) = " << extensions.size() << std::endl;

        for (auto& ext : extensions) {
            std::cout << "[info] ext = " << ext << std::endl;
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        std::cerr << "[warn] validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    void createInstance()
    {

        if (enableValidation && !checkValidationLayerSupport()) {
            throw std::runtime_error("[error] validation layers requested, none found!");
        }

        VkApplicationInfo appInfo {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "hello-vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3
        };

        VkInstanceCreateInfo createInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = this->getRequiredExtensions();

        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo {};
        if (enableValidation) {
            populateDebugMessengerCreateInfo(messengerCreateInfo);

            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&messengerCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateInstance(&createInfo, nullptr, &this->instance) != VK_SUCCESS) {
            throw std::runtime_error("[error] failed to create vk instance.");
        }

        std::cout << "[info] vk instance successfully created." << std::endl;
    }
};

int main()
{
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}