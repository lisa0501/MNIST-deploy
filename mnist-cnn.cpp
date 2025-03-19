//  编译命令：g++ -std=c++14 -I include -L lib -o mnist-onnx mnist-cnn.cpp -lonnxruntime
//  运行命令：./mnist-cnn

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main()
{
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
    std::vector<float> input_tensor_values(1 * 1 * 28 * 28, 0.0f); // 假设输入是一个28x28的灰度图像
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

    return 0;
}
