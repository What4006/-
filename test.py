import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ================= 配置 =================
CONFIG = {
    'data_path': './dataset',
    'model_name': 'resnet50',
    'batch_size': 256,
    'num_workers': 16,
    'num_classes': 36,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 【修改这里】填入你刚才生成的 .pth 文件名
    'ckpt_path': 'best_model_resnet50_20251210-113156.pth' 
}

def main():
    print(f"当前使用设备: {CONFIG['device']}")
    
    # 1. 准备测试集数据
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(CONFIG['data_path'], 'test')
    if not os.path.exists(test_dir):
        print(f"❌ 错误: 找不到测试集文件夹 {test_dir}")
        return

    test_dataset = datasets.ImageFolder(test_dir, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=CONFIG['num_workers'])
    class_names = test_dataset.classes
    print(f"检测到类别数: {len(class_names)}")

    # 2. 初始化模型架构 (必须和训练时一模一样)
    print(f"正在加载模型架构: {CONFIG['model_name']}...")
    if CONFIG['model_name'] == 'resnet50':
        model = models.resnet50(pretrained=False) # 这里不需要预训练权重了，因为我们要加载自己的
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(CONFIG['device'])

    # 3. 加载你训练好的权重
    print(f"正在加载权重文件: {CONFIG['ckpt_path']}...")
    if os.path.exists(CONFIG['ckpt_path']):
        model.load_state_dict(torch.load(CONFIG['ckpt_path']))
        print("✅ 权重加载成功！")
    else:
        print("❌ 错误: 找不到权重文件，请检查文件名！")
        return

    # 4. 开始推理
    model.eval()
    all_preds = []
    all_labels = []

    print("正在测试集上进行推理...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 计算指标
    test_acc = accuracy_score(all_labels, all_preds)
    print(f'>>> 最终测试集准确率 (Test Accuracy): {test_acc:.4f}')

    # 6. 画混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({CONFIG["model_name"]})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    
    save_name = f'confusion_matrix_{CONFIG["model_name"]}_test.png'
    plt.savefig(save_name)
    print(f"混淆矩阵已保存为: {save_name}")

if __name__ == '__main__':
    main()