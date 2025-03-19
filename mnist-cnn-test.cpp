// 使用如下命令编译
// g++ -std=c++14 -I include -L lib -o mnist-onnx-test mnist-cnn-test.cpp -lonnxruntime -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio

// 使用如下命令运行
// ./mnist-onnx-test <image_path>
// 例如：./mnist-onnx-test data/5.png

#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath> // for std::pow and std::sqrt

// 函数：对 float* 数组执行 argmax 操作
size_t argmax(float *data, size_t length)
{
    if (length == 0)
    {
        throw std::invalid_argument("Array length must be greater than 0.");
    }

    size_t max_index = 0;
    float max_value = data[0];

    for (size_t i = 1; i < length; ++i)
    {
        if (data[i] > max_value)
        {
            max_value = data[i];
            max_index = i;
        }
    }

    return max_index;
}

// 计算均值
double calculateMean(const std::vector<float> &data)
{
    double sum = 0.0;
    for (const auto &value : data)
    {
        sum += value;
    }
    return sum / data.size();
}

// 计算标准差
double calculateStandardDeviation(const std::vector<float> &data)
{
    double mean = calculateMean(data); // 先计算均值
    double variance = 0.0;

    for (const auto &value : data)
    {
        variance += std::pow(value - mean, 2); // 计算每个元素与均值的差的平方
    }

    double stdDev = std::sqrt(variance / data.size()); // 标准差是方差的平方根
    return stdDev;
}

// 函数：将图像转换为灰度值并缩放至 28x28，返回 std::vector<float>
std::vector<float> preprocessImage(const std::string &imagePath)
{
    // 读取图像
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return {};
    }

    // 将图像转换为灰度图
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 将图像缩放至 28x28 尺寸
    cv::Mat resizedImage;
    cv::resize(grayImage, resizedImage, cv::Size(28, 28));

    // 将图像数据转换为 std::vector<float>
    std::vector<float> imageData;
    imageData.reserve(28 * 28); // 预分配空间

    for (int y = 0; y < resizedImage.rows; ++y)
    {
        for (int x = 0; x < resizedImage.cols; ++x)
        {
            // 获取像素值并归一化到 [0, 1] 范围
            float pixelValue = resizedImage.at<uchar>(y, x) / 255.0f;
            imageData.push_back(pixelValue);
        }
    }

    // 计算图像的均值和标准差
    double mean = calculateMean(imageData);
    double stdDev = calculateStandardDeviation(imageData);

    // 对每个元素进行正则化
    for (auto &value : imageData)
    {
        value = (value - mean) / stdDev;
    }

    return imageData;
}

int main(int argc, char *argv[])
{
    // 检查是否有输入参数
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_string>" << std::endl;
        return 1; // 返回错误码
    }

    // 初始化ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 加载ONNX模型
    const char *model_path = "mnist-cnn.onnx";
    Ort::Session session(env, model_path, session_options);

    // 获取模型输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    std::vector<const char *> input_node_names(num_input_nodes);
    std::vector<const char *> output_node_names(num_output_nodes);

    for (size_t i = 0; i < num_input_nodes; i++)
    {
        char *input_name = session.GetInputName(i, allocator);
        input_node_names[i] = input_name;
    }

    for (size_t i = 0; i < num_output_nodes; i++)
    {
        char *output_name = session.GetOutputName(i, allocator);
        output_node_names[i] = output_name;
    }

    // 准备输入数据
    // std::vector<float> input_tensor_values(1 * 1 * 28 * 28, 0.0f);  // 假设输入是一个28x28的灰度图像

    // 图像路径
    // 获取输入的字符串参数
    std::string imagePath = argv[1];

    // 预处理图像
    std::vector<float> input_tensor_values = preprocessImage(imagePath);

    std::vector<int64_t> input_tensor_shape = {1, 1, 28, 28};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

    // 运行模型
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_output_nodes);

    // 获取输出结果
    float *output = output_tensors[0].GetTensorMutableData<float>();

    // 10种分类的概率
    std::cout << "Output: ";
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // 执行 argmax 操作，获取分类标签
    size_t max_index = argmax(output, 10);
    std::cout << "Predicted digit: " << max_index << std::endl;

    return 0;
}
