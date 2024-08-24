# ### Streamlit应用程序开发

# %%
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 载入模型
model = joblib.load('XGBoost.pkl')

# 定义特征选项
cp_options = {
    1: '典型心绞痛 (1)',
    2: '非典型心绞痛 (2)',
    3: '非心绞痛 (3)',
    4: '无症状 (4)'
}

restecg_options = {
    0: '正常 (0)',
    1: 'ST-T波异常 (1)',
    2: '左心室肥大 (2)'
}

slope_options = {
    1: '上升 (1)',
    2: '平坦 (2)',
    3: '下降 (3)'
}

thal_options = {
    1: '正常 (1)',
    2: '有过心肌梗塞或其他无法逆转的心脏损伤 (2)',
    3: '某些条件下心脏功能出现异常，但这种异常在休息或其他情况下可以恢复 (3)'
}

# 定义特征名称
feature_names = [
    "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",
    "ST Depression", "Slope", "Number of Vessels", "Thal"
]

# 使用者界面
st.title("基于XGBoost的心脏病预测算法")

# 年龄: 数值型输入
age = st.number_input("年龄:", min_value=1, max_value=120, value=50)

# 性别：类别型输入
sex = st.selectbox("性别 (0=女, 1=男):", options=[0, 1], format_func=lambda x: '女 (0)' if x == 0 else '男 (1)')

# cp: 类别型输入
cp = st.selectbox("胸痛类型:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# trestbps: 数值型输入
trestbps = st.number_input("静息血压 (trestbps):", min_value=50, max_value=200, value=120)

# chol:数值型输入
chol = st.number_input("血清胆固醇 mg/dl (chol):", min_value=100, max_value=600, value=200)

# fbs: 类别型输入
fbs = st.selectbox("空腹血糖 > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: '不是 (0)' if x == 0 else '是的 (1)')

# restecg: 类别型输入
restecg = st.selectbox("静息心电图(ECG)结果:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

# thalach: 数值型输入
thalach = st.number_input("最大心率 (thalach):", min_value=50, max_value=250, value=150)

# exang: 类别型输入
exang = st.selectbox("运动型心绞痛 (exang):", options=[0, 1], format_func=lambda x: '不是 (0)' if x == 0 else '是的 (1)')

# oldpeak: 数值型输入
oldpeak = st.number_input("运动状态下ST下降的值 (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# slope: 类别型输入
slope = st.selectbox("心电图(ECG)的ST段斜率 (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

# ca: 数值型输入
ca = st.number_input("冠状动脉狭窄的数量 (ca):", min_value=0, max_value=4, value=0)

# thal: 类别型输入
thal = st.selectbox("地中海贫血 (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# 处理输入并做出预测
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

if st.button("点击预测"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class}")
    st.write(f"**预测概率:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"根据我们的模型，您患心脏病的风险很高. "
            f"根据模型预测，您患心脏病的概率为 {probability:.1f}%. "
            "虽然这只是一个估计值，但它表明您可能面临很大的风险. "
            "我建议你尽快去看心脏病医生, "
            "以确保您得到准确的诊断和必要的治疗."
        )
    else:
        advice = (
            f"根据我们的模型，您患心脏病的风险很低. "
            f"根据模型预测，您不患心脏病的概率为 {probability:.1f}%. "
            "然而，保持健康的生活方式仍然非常重要. "
            "我建议您定期进行体检，以监测心脏健康状况, "
            "并在出现任何症状时及时就医."
        )

    st.write(advice)

    # 计算 SHAP 值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")


