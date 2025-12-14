import os
import cv2  # OpenCV，处理图像的
import numpy as np # 数学计算，处理矩阵
import matplotlib.pyplot as plt # 画图工具
import seaborn as sns # 基于 matplotlib，让图更好看（画热力图用）
from time import time # 计时用，看看训练花了多久

# 下面是 sklearn 全家桶
from sklearn.model_selection import train_test_split, GridSearchCV # 数据切分、网格搜索
from sklearn.decomposition import PCA # 降维算法
from sklearn.svm import SVC # 支持向量机分类器
from sklearn.preprocessing import StandardScaler # 数据标准化
from sklearn.pipeline import Pipeline # 管道，把上面几个步骤串起来
from sklearn.metrics import classification_report, confusion_matrix # 评分报告

Dataset_Path = "D:/moshishibie_homework/dataset/train"
IMG_SIZE = 64

def load_data(dataset_path):
    images=[]
    labels=[]
    categories=os.listdir(dataset_path)

    for category in categories:
        folder_path=os.path.join(dataset_path,category)
        if not os.path.isdir(folder_path): continue

        for file in os.listdir(folder_path):
            img_path=os.path.join(folder_path,file)

            try:
                img=cv2.imread(img_path)
                img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
                img_flat=img.flatten()
                images.append(img_flat)
                labels.append(category)

            except Exception as e:
                pass

    return np.array(images), np.array(labels), categories

x,y,class_names=load_data(Dataset_Path)
print(f"数据加载: {x.shape[0]} 张图片, 每张图片 {x.shape[1]} 维特征")

x_train,x_test,y_train,y_test=train_test_split(test_size=0.2,random_state=42)

pipe=Pipeline([
    ('scaler',StandardScaler()),
    ('pca',PCA(n_components=0.95)),
    ('svc',SVC(kernel='rbf', class_weight='balanced'))
#SVM的损失函数是内置的Hinge loss函数
#优化器在SVM里不存在，但是可以使用求解SVM的对偶问题来进行优化
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],       # 正则化系数
    'svm__gamma': [0.0001, 0.001, 0.01] # 核函数系数
}

print(f"开始 {5} 折交叉验证与网格搜索 (PCA+SVM)... 这可能需要几分钟")
t0 = time()

# 使用5折叠而不使用10折叠的原因是
# 1.采用10折叠使得每次的训练集较大导致每次划分的训练集都比较接近，使得每次训练的内容都更接近，导致每次训练得出的结果都更接近使得模型的结果更完美，但是实际上可能是它们都犯了一样的错误
# 2.10折叠使得验证集较小，导致一次训练时，刚好抽中的验证集大部分都是模型擅长识别的类别，使得不能很好的反映模型的真实情况，降低了模型的泛化能力
grid = GridSearchCV(pipe, param_grid, cv=5, verbose=2, n_jobs=-1)
grid.fit(x_train, y_train)

print(f"训练完成，耗时: {time() - t0:.2f}s")
print(f"最优参数组合: {grid.best_params_}") 
print(f"交叉验证最高准确率: {grid.best_score_:.4f}")

y_pred = grid.predict(X_test)

print("\n分类报告:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (PCA + SVM)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show() 

results = grid.cv_results_
scores = results['mean_test_score'].reshape(len(param_grid['svm__C']), len(param_grid['svm__gamma']))

plt.figure(figsize=(8, 6))
sns.heatmap(scores, annot=True, fmt=".3f", cmap="viridis",
            xticklabels=param_grid['svm__gamma'],
            yticklabels=param_grid['svm__C'])
plt.title('Validation Accuracy: C vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('C Parameter')
plt.show() 

n_components = grid.best_estimator_.named_steps['pca'].n_components_
print(f"PCA 将特征从 {X.shape[1]} 维降低到了 {n_components} 维")