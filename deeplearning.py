import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
import copy

# ================= 1. 超参数与环境配置 (CONFIG) =================
CONFIG = {
    'data_path': './dataset', 
    'model_name': 'resnet50',      
    'batch_size': 256,             
    'num_workers': 16,             
    'learning_rate': 0.01,        
    'num_epochs': 20,              
    'num_classes': 36,             
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    print(f"当前使用设备: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True 

    # ================= 2. 数据增强与预处理 =================
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 【新增】测试集预处理（通常和验证集一样，不做增强）
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # ================= 3. 加载数据集 (Train & Val) =================
    print("正在加载训练与验证数据集...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(CONFIG['data_path'], x),
                                              data_transforms[x])
                      for x in ['train', 'validation']}
    
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=CONFIG['batch_size'],
                                 shuffle=True, 
                                 num_workers=CONFIG['num_workers'], 
                                 pin_memory=True) 
                   for x in ['train', 'validation']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes
    print(f"检测到类别数: {len(class_names)}")
    print(f"训练集: {dataset_sizes['train']} 张, 验证集: {dataset_sizes['validation']} 张")

    # ================= 4. 模型构建 =================
    print(f"正在初始化模型: {CONFIG['model_name']}...")
    if CONFIG['model_name'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif CONFIG['model_name'] == 'resnet50':
        model = models.resnet50(pretrained=True)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(CONFIG['device'])

    # ================= 5. 损失函数与优化器 =================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate']) 
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ================= 6. 训练循环 =================
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"开始训练 (Batch Size: {CONFIG['batch_size']})...")
    for epoch in range(CONFIG['num_epochs']):
        print(f'Epoch {epoch+1}/{CONFIG["num_epochs"]}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
                
                # 深拷贝保存目前最好的模型参数
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最优验证集准确率: {best_acc:4f}')

    # ================= 7. 结果可视化与保存 =================
    # 保存结果图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title(f'Loss ({CONFIG["model_name"]})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.title(f'Acc ({CONFIG["model_name"]})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_name_png = f'result_{CONFIG["model_name"]}_bs{CONFIG["batch_size"]}_{timestamp}.png'
    plt.savefig(save_name_png)
    print(f"结果图已保存为: {save_name_png}")

    # 【新增功能 1】保存最佳模型权重
    model_save_name = f'best_model_{CONFIG["model_name"]}_{timestamp}.pth'
    torch.save(best_model_wts, model_save_name)
    print(f"最佳模型权重已保存为: {model_save_name}")

    # ================= 8. 【新增功能 2】在测试集上评估 =================
    print("\n" + "="*20 + " 开始测试集评估与可视化 " + "="*20)
    test_dir = os.path.join(CONFIG['data_path'], 'test')
    
    if os.path.exists(test_dir):
        # 1. 准备测试数据
        test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                                 shuffle=False, num_workers=CONFIG['num_workers'])
        
        # 2. 加载最佳权重
        model.load_state_dict(best_model_wts)
        model.eval()

        all_preds = []
        all_labels = []

        # 3. 推理 (这次我们要收集所有的预测结果，用来画图)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # 收集预测结果和真实标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 4. 计算准确率
        from sklearn.metrics import accuracy_score, confusion_matrix
        import seaborn as sns
        import numpy as np

        test_acc = accuracy_score(all_labels, all_preds)
        print(f'>>> 最终测试集准确率 (Test Accuracy): {test_acc:.4f}')

        # 5. 绘制混淆矩阵 (High Score Alert!)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10)) # 画布调大一点，因为有36个类
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix ({CONFIG["model_name"]} - Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45) # 类别名字旋转一下，防止重叠
        
        # 保存混淆矩阵图
        cm_save_name = f'confusion_matrix_test_{CONFIG["model_name"]}_{timestamp}.png'
        plt.savefig(cm_save_name)
        print(f"测试集混淆矩阵已保存为: {cm_save_name}")
        
    else:
        print(f"警告: 未找到测试集文件夹 '{test_dir}'")

if __name__ == '__main__':
    main()