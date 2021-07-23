import pandas as pd

# 새로운 데이터를 구해왔음
train_df = pd.read_csv('input/train.csv')
train_df.info() # -> 데이터는 500행, 11열

"""
================= 데이터에 대한 설명 ==================
( 데이터 출저 : https://www.kaggle.com/ajaymanwani/loan-approval-prediction/data )
( 코드 출처 : https://www.kaggle.com/ninzaami/loan-predication )

 rows : 614
 columns : 13
  
 Loan_ID : 대출자들을 구별하는 식별자
 Loan_Status : Loan approved(Y/N)
 Principal : 대출 금액
 Dependents : 부양 가족수 (예: 0, 1, 2, 3+)
 Self_Employed : 자영업자 여부
 ApplicantIncome : 대출 신청인 소득
 CoapplicantIncome : 공동 신청자 소득
 LoanAmount : 대출 금액
 Loan_Amount_Term : 만기 기간
 Credit_History : 신용 내역이 기준에 적합한지 여부 ( 1 : 적합, 0: 미달 )
 Property_Area : 거주 지역 ( Urban, Semi Urban, Rural )
 Age, Gender, Education, Married : 고객에 대한 기본 정보
 
 정보를 보면 결측치가 있는것을 확인할 수 있다. ( 예: LoanAmount 변수의 경우 null값이 아닌 것이 614개 중에서 592이므로 나머지 22개의 결측치가 있음을 알 수 있다.)
 ==================================================
"""

""" 
 =================== 변수 타입별로 분류 ======================
# 변수 타입에는 크게 4가지가 있다. ( 이진형, 범주형, 정수형, 연속형 )
# 이진형 : Genger(M,F), Married(Yes,No), Education(Graduate, Not Graduate), Self_Employed(Yes, No), Loan_Status(Y, N) - 2가지로 표현되는 변수
# 범주형 : Dependents(0, 1, 2, 3+), Property_Area(Urban, Semiurban, Rural), Loan_Amount_Term(120, 240, 360) - 2가지 이상으로 표현되는 변수
# 정수형 : ApplicantIncome, CoapplicantIncome, LoanAmount - 정수형 숫자로 표현되는 변수

분류한 이유를 알 수 있을까? 책에 따른다면 알고리즘에 적합한 타입들을 알기 위해서 타입을 구별할 줄 알아야한다고 적혀있다. 
정확한 이유를 알 수 있다면 후에 서술해두어야겠다.
 ==========================================================
"""

# 여기서는 크게 범주형과 정수형으로 나누었다.
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
print(categorical_columns)
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print(numerical_columns)


# 하원이 말에 따르면 import는 라이브러리 전체를, from은 라이브러리 중 일부만을 가져오는 식으로 쓰인다고 했다.
import seaborn as sns
import matplotlib.pyplot as plt


"""
================== subplotlib 에 대한 설명 ===========================
matplotlib는 크게 3가지객체 (FigureCanvas, Renderer, Artist) 로 구성되어있다.
FigureCanvas : 그림 그릴 영역을 나타내는 객체
Renderer : FigureCanvas에 그리는 도구 객체
Artist : Renderer가 FigureCanvas에 어떻게 그릴 것인가를 나타내는 객체
화가가 그림을 그리는 모습을 상상한다면 각 객체 역할을 쉽게 이해할 수 있다.

보통의 사용자는 Artist 객체애 대해서만 생각하면 된다고 한다.

Aritist는 크게 2가지(Primitives, Containers)로 분류된다.
Primitives : Line2D, Rectangle, Text, AxesImage, Patch 등과 같이 캔버스에 그려지는 표준 그래픽 객채
Containers : Axis, Axes, Figure 등과 같이 이들 Primitives가 위치하게 될 대상

Figure : 도화지
Axes : 도화지내에 그림이 그려질 공간 ( Figure와 Axces 사이에는 여백이 있음 )
Axis : Axes 안에서 x축, y축을 나타냄
Line2D : Axes안에서 실제 그림이 그려지는 공간

maplotlib 구동 방식이 크게 2가지로 있다.
1. pyplot API 사용 : 가져온 pyplot을 그대로 사용하는것? ( 예 : plt.subplots() )
2. pyplot 객체 사용 : pyplot으로 객체를 만들고 객체를 이용하는 방법 ( 예 : fig = plt.figure() )
3. 1,2를 적절하게 섞는 방법

plt.subplots()함수는 figure 객체를 생성하고 figure.subplots()를 호출하여 리턴
만약 figure, axes = plt.subplots(4,2)라고 하면 4행 2열로 총 8개의 axes가 만들어지고 2차원 배열로 axes 변수에 저장된다.
===================================================================
"""

# subplots는 한 화면에 여러개의 그래프를 그릴때 사용.
fig, axes = plt.subplots(4,2,figsize=(12,15))  # fig는 figure 객체를 의미, axes는 그래프 각각을 나태는 객체정도로 볼 수 있겠다. ( 이 방식은 처음 본다. )
# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)
# 보통 이런식으로 많이 보았는데,
# mapplotlib에 대한 자세한 내용은 링크를 참고하자. ( https://wikidocs.net/14604 )

for idx, cat_col in enumerate(categorical_columns): # enumerate는 인덱스와 배열 원소를 같이 반환한다. (몇번째 반복문인지 확인할 때 필요하다. )
    row, col = idx//2, idx%2
    sns.countplot(x=cat_col, data=train_df, hue='Loan_Status', ax=axes[row, col])

plt.subplots_adjust(hspace=1) # 자세한 내용은 모르겠다. 귀찮으니까 일단 넘기기로 했다.
plt.show() # 안 쓰면 안보인다.

# 결과를 보면 신용 대출에 영향을 미친다고 판단되는 변수들을 확인할 수 있다.
# 내 생각에는 Dependents(0), Married(N), Education(graduate), Credit_History(Y), Loan_Amount_Term(360) 정도가 정확한것 같고, Property_Area(Semiurban)의 경우는 결과가 이해가 안된다.

fig, axes = plt.subplots(1,3,figsize=(17,5))
for idx, cat_col in enumerate(numerical_columns):
    sns.boxplot(y=cat_col, data=train_df, x='Loan_Status', ax=axes[idx])

print(train_df[numerical_columns].describe())
plt.subplots_adjust(hspace=1)
plt.show()

# 정수형 변수들은 박스형태로 보는것이 더 좋은가?
# 정수형 변수들에서는 딱히 결정적인 관계를 나타내는것을 찾지는 못하겠다.


""" 
================ 범주형 데이터 엔코딩 ========================  
컴퓨터는 숫자를 이해할 수 있어서 문자를 수치적으로 바꾸는 작업이 필요하다. 이 작업을 엔코딩이라고 하는것 같다.
여기서 가변수라는 개념이 등장한다. 범주형 데이터를 단순히 수치적으로 나타내게 된다면 의도하지 않은 관계성이 생길 수 있다.
예를 들면 요일이라는 변수에 월요일은 1, 화요일은 2 이런식으로 쓰게 된다면 4.5요일에 어떤 특징이 두드러진다같은 있을 수 없는 결론을 얻을지도 모른다.
따라서 범주형 각 값을 변수로 만들고 해당 변수를 0과1을 통해서 어떤 요일에 해당하는지로 나타낸는 개념이라고 보면 된다. ( https://devuna.tistory.com/67 )
==========================================================
"""

print(train_df.head())
train_df_encoded = pd.get_dummies(train_df, drop_first=True) # 결측치를 제외한 0과1로 구성된 더미값이 만들어진다. 옵션(dummy_na=True)를 설정하면 결측값도 더미값이된다.
print(train_df_encoded.head())

# 독립변수와 종속변수를 각가 X, y로 설정
X = train_df_encoded.drop(columns='Loan_Status_Y') # 더미값이 '변수명_값' 형식으로 생성되는것 같다.
y = train_df_encoded['Loan_Status_Y']

# 학습 데이터와 평가 데이터를 나누어주는 함수 호출
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y, random_state=42)

"""
=============== 결측치 대체(imputation) ==========================

불완전한 데이터셋은 사용하기 힘들다. 따라서 불완전한 부분을 완전하게 만들어줘야한다.
이때 사용하는 방법 중 가장 간단한 것은 결측치가 포함된 열과 행을 버리는 것이다. 그러나 이 방법은 중요한 변수를 희생할수도 있다는 것이다.
그래서 더 나은 방법으로 값을 대체한다. 

대체 알고리즘은 크게 2가지가 있는데, 
하나는 한 변수에서 결측치가 아닌 값들을 이용해서 결측치 값을 추정하는것이고 ( univariate imputation )
다른 하나는 사용할 수 있는 여러 변수값들을 이용해서 해당 변수의 결측치를 추정하는 것이다. ( multivariate imputation )
여기서 사용하는 SimpleImputer는 univariate imputation에 해당된다. ( 출저 : https://scikit-learn.org/stable/modules/impute.html#impute )

univariate imputation은 지정된 값 혹은 결측치 변수의 통계값(mean, median, most frequent and so on)으로 대체할 수 있다. 
================================================================
"""

# 결측치는 각 해당 변수의 평균값으로 대체
from sklearn.impute import SimpleImputer # impute는 결측치를 특정값으로 대체하는 방법
imp = SimpleImputer(strategy='mean') # 평균값으로 대체해줄 imputer 선언 ( 어떤 값으로 대체하는지에 대한 기준은 잘 모르겠다. )
imp_train = imp.fit(X_train)
X_train = imp_train.transform(X_train)
X_test_imp = imp_train.transform(X_test)

# 모델 생성 : decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score # sklearn.model_selection은 데이터를 나누고 검증할 때 사용되는 모듈이다.
from sklearn.metrics import accuracy_score, f1_score #sklearn.metrics은 scikit-learn 패키지 중에서 모델 평가에 사용되는 모듈이다.

"""
==================== 의사결정트리 알고리즘(decision tree algorithm) ===========================
데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 트리 기반의 분류 규칙을 만드는 알고리즘이다.
(if, else 문등을 이용하는 것처럼 이용해서 규칙을 만들고 예측하는거라고 생각하면 된다. )

장점 : 쉽고 직관적이다, 각 변수의 스케일이나 정규화에 영향을 크게 받지 않는다.
단점 : 규칙이 많아질수록 모델이 복잡해지고 과적합이 일어난다. 그래서 트리의 크기를 사전에 제한하는 튜닝이 필요하다.
===========================================================================================
"""

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_train)
print("Training Data Set Accuracy: ", accuracy_score(y_train, y_pred))
print("Training Data F1 Score: ", f1_score(y_train, y_pred))

"""
======================== 알고리즘 성능 평가 ==============================
머신러닝 알고리즘을 작성했다면 이제 얼마나 잘 작동하는지 확인하는 작업이 필요하다.
평가 지표가 여러개가 있는데, Accuracy(정확도), Recall(재현율), Precision(정밀도), F1 Score 가 대표적이다.

평가시 데이터를 4가지로 분류할 수 있다. (True positive, True negative, False positive, False negative )
True : 예측이 성공한 경우
False : 예측이 실패한 경우
Positive : 기준값 이상이라고 예측한 경우
Negative : 기준값 미만이라고 예측한 경우

Accuracy : 전체 데이터 중에서 올바르게 분류한 데이터 수 ( True / All )
Recall : 

"""


# 결과 평가
print("Validation Mean F1 Score : ", cross_val_score(tree_clf, X_train, y_train, cv=5, scoring='f1_macro').mean())
print("Validation Mean Accuracy : ", cross_val_score(tree_clf, X_train, y_train, cv=5, scoring='accuracy').mean())

