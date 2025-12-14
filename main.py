import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from time import time

# =================配置区域=================
# 根据文件 [cite: 68] 下载的数据集路径
DATASET_PATH = "D:/moshishibie_homework/dataset/train"  # 假设你的路径
IMG_SIZE = 64  # 将图片统一缩放到 64x64，降低初始计算量
RANDOM_STATE = 42
KFOLDS = 5     # K折交叉验证的 K 值 [cite: 49]

def load_data(path):
    print("正在加载图片数据...")
    images = []
    labels = []
    categories = os.listdir(path)
    
    for category in categories:
        folder_path = os.path.join(path, category)
        label = category
        if not os.path.isdir(folder_path): continue
        
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                # 读取并转灰度（PCA通常处理单通道更高效，除非颜色特征极其重要）
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # 展平图像: (64, 64, 3) -> 12288 维向量
                img_flat = img.flatten()
                images.append(img_flat)
                labels.append(label)
            except Exception as e:
                pass
                
    return np.array(images), np.array(labels), categories

# 1. 加载数据
X, y, class_names = load_data(DATASET_PATH)
print(f"数据加载完成: {X.shape[0]} 张图片, 每张图片 {X.shape[1]} 维特征")

# 2. 划分数据集 (Train/Test)
# 虽然有K折验证，但通常还是保留一个独立的 Test Set 用于最终评估
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# =================核心算法实现: PCA + SVM + K-Fold=================
# 使用 Pipeline 串联步骤，确保代码规范，防止验证集信息泄露
# 步骤: 标准化 -> PCA降维 -> SVM分类
pipe = Pipeline([
    ('scaler', StandardScaler()),        # SVM 对数据尺度敏感，必须标准化
    ('pca', PCA(n_components=0.95)),     # 保留 95% 的方差，维数自动确定
    ('svm', SVC(kernel='rbf', class_weight='balanced')) # RBF核处理非线性
])

# 定义超参数网格 (Grid Search)
# 这里对应文件要求：调整超参数如C值、核函数等 [cite: 72]
param_grid = {
    'svm__C': [0.1, 1, 10, 100],       # 正则化系数
    'svm__gamma': [0.0001, 0.001, 0.01] # 核函数系数
}

print(f"开始 {KFOLDS} 折交叉验证与网格搜索 (PCA+SVM)... 这可能需要几分钟")
t0 = time()

# GridSearchCV 自动执行 K-Fold
grid = GridSearchCV(pipe, param_grid, cv=KFOLDS, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"训练完成，耗时: {time() - t0:.2f}s")
print(f"最优参数组合: {grid.best_params_}") # 这就是你要写进报告的数据
print(f"交叉验证最高准确率: {grid.best_score_:.4f}")

# =================模型评估与图表生成=================
# 3. 在测试集上评估
y_pred = grid.predict(X_test)

print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 4. 生成高分图表 1: 混淆矩阵 (Confusion Matrix)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (PCA + SVM)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show() # 截图放进报告

# 5. 生成高分图表 2: 超参数性能热力图 (对应文件要求列出不同超参数下的结果表 )
results = grid.cv_results_
scores = results['mean_test_score'].reshape(len(param_grid['svm__C']), len(param_grid['svm__gamma']))

plt.figure(figsize=(8, 6))
sns.heatmap(scores, annot=True, fmt=".3f", cmap="viridis",
            xticklabels=param_grid['svm__gamma'],
            yticklabels=param_grid['svm__C'])
plt.title('Validation Accuracy: C vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('C Parameter')
plt.show() # 截图放进报告，这展示了你的调优过程

# 6. 查看 PCA 降维后的维度
n_components = grid.best_estimator_.named_steps['pca'].n_components_
print(f"PCA 将特征从 {X.shape[1]} 维降低到了 {n_components} 维")

"""
训练完成，耗时: 207.87s
最优参数组合: {'svm__C': 10, 'svm__gamma': 0.0001}
交叉验证最高准确率: 0.3182

分类报告:
               precision    recall  f1-score   support

        apple       0.13      0.11      0.12        18
       banana       0.70      0.44      0.54        16
     beetroot       0.37      0.45      0.41        22
  bell pepper       0.18      0.14      0.16        21
      cabbage       0.39      0.84      0.53        19
     capsicum       0.33      0.17      0.22        24
       carrot       0.70      0.70      0.70        20
  cauliflower       0.53      0.60      0.56        15
chilli pepper       0.29      0.22      0.25        18
         corn       0.31      0.27      0.29        15
     cucumber       0.22      0.33      0.27        18
     eggplant       0.21      0.27      0.24        11
       garlic       0.38      0.67      0.48        18
       ginger       0.33      0.33      0.33         9
       grapes       0.71      0.50      0.59        30
     jalepeno       0.43      0.19      0.26        16
         kiwi       0.62      0.44      0.52        18
        lemon       0.62      0.62      0.62        16
      lettuce       0.46      0.52      0.49        21
        mango       0.19      0.19      0.19        16
        onion       0.22      0.19      0.21        21
       orange       0.36      0.36      0.36        11
      paprika       0.29      0.45      0.36        11
         pear       0.55      0.38      0.44        16
         peas       0.38      0.20      0.26        15
    pineapple       0.46      0.33      0.39        18
  pomegranate       0.38      0.43      0.40        14
       potato       0.31      0.25      0.28        16
      raddish       0.42      0.31      0.36        16
    soy beans       0.32      0.25      0.28        24
      spinach       0.19      0.21      0.20        19
    sweetcorn       0.20      0.31      0.24        13
  sweetpotato       0.25      0.27      0.26        15
       tomato       0.30      0.53      0.38        19
       turnip       0.35      0.35      0.35        20
   watermelon       0.10      0.07      0.08        14

     accuracy                           0.36       623
    macro avg       0.37      0.36      0.35       623
 weighted avg       0.38      0.36      0.36       623
"""