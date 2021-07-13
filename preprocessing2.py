import pandas as pd

train_df = pd.read_csv('input/train3.csv')
train_df.info()
"""
======================= 데이터 정보 ========================
출저 : https://www.kaggle.com/teertha/personal-loan-modeling

rows : 5000
columns : 14

ID : Customer ID
Age : Customer age
Experience : Number of years of prefessional experience
Income : Annual income 
ZIP Code : Home Address ZIP code
Family : Family size of teh customer
CCAvg : Avg. spending on credit card per month
Education : Education Level ( ex: 1: Undergread, 2: Graduate, 3: Advanced/Professional )
Mortgage : Value of house ( if any 0$ )
Personal Loan : Did this customer accept the personal loan offered in the last campaign?
CreditCard : Dose the customer use a credit card issued by this Bank? ( 0 : No, 1 : Yes )

"""

# ============================== 대출 승인 여부를 결정하는 특징들을 선별 ==================================
# 필요한 열을 골라내는 작업이 필요해보인다. 대출 승인을 예측하는데 필요한 특징들이 어떤 것들이 있을지 조사해야한다.
# 대출 승인 기준은 여기를 참고하였다. ( https://blog.peoplefund.co.kr/order-of-credit-verification/ )
# 블로그의 내용에 따르면 대출 승인 여부는 대출 신청인의 소득(Income), 직업 안정성(Experience), 대출 규모, 신용점수(CreditCard)와 높은 관련이 있는것 같다
# 데이터가 항상 내가 원하는 특징을 갖을 가능성은 낮은것 같다. 찾아보아도 위의 특징을 잘 만족하면서도 다루기 쉬운 데이터를 찾는것은 불가능했고 지금의 데이터를 찾는것만 해도 제법 시간이 걸렸다.
# 대신 가지고 있는 특징들을 어떻게 적절히 활용할 수 있을지를 생각해봐야겠다.

# 신용점수같은 경우 CreditCard, CCAvg변수를 이용하면 어느정도 보완할 수 있을 것 같다.
# link : https://www.edaily.co.kr/news/read?newsId=03460406622682440&mediaCodeNo=257
# 신용카드 사용 이력은 은행이 대출 승인을 결정하는데 필요한 정보를 제공해준다.
# 남겨진 사용 이력을 보고 제대로 사용한 기록이 남아있는 경우가 기록이 없는 경우보다 신용 등급이 높게 평가된다고 한다. 한도 초과하는 경우가 포함될 수 있지만 대부분의 사람은 제대로 지킬것이라 가정한다.


# 대출 규모의 경우는 Income, Mortage, CCAvg로 어느정도 짐작할 수 있을 것 같다.
# 대체로 수익과 부동산에 따라 사람의 경제 수준을 알 수 있고 경제 수준이 높은 사람일수록 빌리는 액수가 클 것이라 예상한다.


# 다른 변수(Family, Education) 같은 변수들도 활용처가 있을법도 한데 일단은 배제하고 진행하겠다.

# ===================================== 데이터 분석 방법 선정 ==========================================
# 분석 방법은 다중선형회귀분석으로 선택했다.
# 다중회귀분석이란? 두 개 시앙의 독립변수들과 하나의 종속변수의 관계를 분석하는 방법으로 위에서 골라낸 여러개의 변수들과 대출여부 변수(Personal Loan) 사이의 관계를 분석하는 방법
# 독립변수 : Experience, Income, CCAvg, Creditcard
# 종속변수 : Personal Loan

# 제발이번에는 되었으면 좋겠다.