import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report

data = pd.read_csv('结果.csv')
y_pred = []
y_true = []

for x, y in zip(data['预测'], data['真实']):
    y_pred.append(0 if x == 'cat' else 1)
    y_true.append(0 if y == 'cat' else 1)

print(classification_report(y_true, y_pred))
