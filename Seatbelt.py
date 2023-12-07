#-------------------------------------------------------------
import pandas as pd
import seaborn as sns

data = pd.read_csv('./data/train-new.csv')

sns.barplot(data=data,x='seatbelt',y='dead')
sns.boxplot(x='injSeverity',y='dead',data=data,hue='seatbelt')
sns.barplot(data=data,x='dead',y='injSeverity')
#-------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 데이터 불러오기
data = pd.read_csv("./data/train-new.csv")

# 전처리 단계
data['seatbelt'] = data['seatbelt'].map({'none': 0, 'belted': 1})
data['weight'].fillna(data['weight'].mean(), inplace=True)
data = data[data['weight'] < 500]
data['dead'] = data['dead'].map({'dead': 0, 'alive': 1})

# Feature와 Target 설정
X = data[['seatbelt']]
y = data['dead']

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# 예측
y_pred_logreg = logreg_model.predict(X_test)

# 정확도 평가
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f'Accuracy (Logistic Regression): {accuracy_logreg:.2f}')

# 로지스틱 회귀에 대한 예측 확률 얻기
y_prob_logreg = logreg_model.predict_proba(X_test)[:, 1]

# ROC 곡선 생성
fpr, tpr, thresholds = roc_curve(y_test, y_prob_logreg)

# AUC 계산
auc = roc_auc_score(y_test, y_prob_logreg)

# ROC 곡선 시각화
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show()

# 분류 보고서 출력
print(classification_report(y_test, y_pred_logreg))
