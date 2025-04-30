import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from sklearn.preprocessing import MinMaxScaler

# 데이터 로딩
@st.cache
def load_data():
    # 예시 데이터프레임
    data = pd.read_csv('your_data.csv')  # 예시로 CSV 파일을 로드합니다
    return data

data = load_data()

# 1. 데이터 탐색
st.title('모델 성능 평가 및 시각화')

st.subheader("데이터 샘플")
st.write(data.head())

# 2. 입력 변수(X)와 목표 변수(y) 설정
X = data[['일평균공급량', '수용가수', '야간최소유량', '야간최소유량비율']]
y = data['wrr']

# 3. 데이터 분할 (60:20:20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. 모델 학습
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. 검증 데이터로 성능 평가
y_val_pred = rf_model.predict(X_val)
mse_val = MSE(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = R2(y_val, y_val_pred)

# 6. 테스트 데이터로 성능 평가
y_test_pred = rf_model.predict(X_test)
mse_test = MSE(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = R2(y_test, y_test_pred)

# 7. 시각화
st.subheader("검증 데이터 성능")
st.write(f"MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.2f}")

# 8. 성능 평가 - 테스트 데이터
st.subheader("테스트 데이터 성능")
st.write(f"MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.2f}")

# 9. 히스토그램 시각화
st.subheader("입력 변수 히스토그램")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# 수치형 변수의 히스토그램을 그리기
numeric_cols = ['일평균공급량', '수용가수', '야간최소유량', '야간최소유량비율']
for i, col in enumerate(numeric_cols):
    axes[i].hist(data[col], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(f'{col} 분포')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('빈도')

st.pyplot(fig)

# 10. 상관 행렬 히트맵
st.subheader("상관 관계 히트맵")
plt.figure(figsize=(8, 6))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
st.pyplot()

# 11. 예측값 시각화 (예: 실제값과 예측값 비교)
st.subheader("실제값 vs 예측값")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('실제값 vs 예측값')
st.pyplot()
