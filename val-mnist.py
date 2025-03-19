import onnxruntime as ort
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 加载测试数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载ONNX模型
onnx_model_path = "mnist-cnn.onnx"
session = ort.InferenceSession(onnx_model_path)

# 获取输入和输出名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 验证模型
correct = 0
total = 0

for i, (images, labels) in enumerate(test_loader):    
    # 将PyTorch张量转换为NumPy数组
    images_np = images.numpy()
    
    # 运行ONNX模型进行推理
    outputs = session.run([output_name], {input_name: images_np})
        
    outputs = np.array(outputs[0])  # 获取输出结果
    
    # 获取预测类别
    predicted = np.argmax(outputs, axis=1)
    
    if i % 100 == 0:
        print(outputs)
        print(predicted)
        print(labels.numpy())
        
    # 统计正确预测的数量
    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum()

# 计算准确率
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%, total: {total}, correct: {correct}')