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
"""


# 열을 보니까 성별은 대출 승인에 영향이 없을 것 같다 -> 불필요한 정보를 없애서 데이터의 크기를 줄여보자


# Gender 열이 줄어들었다. 추가로 더 없앨만한게 있을까 탐색해보고 인터넷 정보를 통해서 기준을 잡아보자.
# 대출 승인 기준은 여기를 참고하였다. ( https://blog.peoplefund.co.kr/order-of-credit-verification/ )
# 블로그의 내용에 따르면 대출 승인 여부는 대출 신청인의 소득(ApplicantIncome, CoapplicantIncome), 직업 안정성, 대출 규모(LoanAmount), 신용점수(Credit_History)과 높은 관련이 있는것 같다
# 해당 데이터에서는 직업 안정성변수는 들어있지 않아서 조금 아쉽지만 그런대로 예측은 될것같다고 느껴진다.

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
print(categorical_columns)
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print(numerical_columns)

# 제발이번에는 되었으면 좋겠다.