import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 加载Excel文件
file_path = 'data_original.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# def count_industry_categories(data, industry_column):
#     """
#     返回:
#         pd.Series: 每个行业分类的出现频率。
#     """
#     if industry_column not in data.columns:
#         raise ValueError(f"列 '{industry_column}' 不存在于数据中。")
#
#     # 统计行业分类的出现频率
#     industry_counts = data[industry_column].value_counts()
#     return industry_counts
#
# industry_counts = count_industry_categories(df, '行业分类')
# # print("行业分类统计结果:")
# # print(industry_counts) # Name: count, Length: 317
# df = pd.get_dummies(df, columns=['行业分类'], prefix='Industry')


# 查看数据
# print(df.dtypes)
# print(df.head())
print(df.columns)
# 清理数据（去除逗号和非数字字符）
df = df.replace({',': '', '--': None}, regex=True)

# 转换为数值类型
df = df.apply(pd.to_numeric, errors='coerce')
# 处理缺失值
df = df.fillna(method='ffill').fillna(method='bfill')

# 将百分比数据转换为小数
percentage_columns = [col for col in df.columns if '率' in col or '费用/营业总收入' in col]
df[percentage_columns] = df[percentage_columns] / 100

# 将数据转换为数值类型（去除逗号）
df = df.replace({',': ''}, regex=True)
df = df.apply(pd.to_numeric, errors='coerce')

# 查看清洗后的数据
print(df.head())
print(df.dtypes)
# 保存处理后的数据到Excel
output_file_path = 'cleaned_data.xlsx'  # 保存路径
df.to_excel(output_file_path, index=False)
print(f"处理后的数据已保存到 {output_file_path}")

def create_dataset(data):
    X, Y = [], []
    # 获取第i家企业的数据
    for i in range(len(data)):
        row_data = data.iloc[i].values  # 第i家企业的数据

        # 将10年的数据拆分为10组
        for year in range(10):
            # 每年的特征（10个特征，跨年提取）
            year_features = []
            for feature_idx in range(10):  # 假设有10个特征
                # 提取跨年特征
                feature_value = row_data[year + feature_idx * 10]  # 跨年提取特征
                year_features.append(feature_value)

            # 加入时间特征
            time_feature = 2014 + year  # 时间因素为年份
            X.append(year_features + [time_feature])  # 加入时间特征

            # 每年的目标值（净利润）
            year_target = row_data[year + 10 * 10]  # 假设净利润存储在每年的第11个位置
            if np.isnan(year_target):
                print(f"第{i}家企业的第{year}年净利润数据缺失。")
                exit(1)
            Y.append(year_target)
    print("X[0]:", X[0])  # 打印第一个样本的特征
    print("Y[0]:", Y[0])  # 打印第一个样本的目标值
    return np.array(X), np.array(Y)

X, Y = create_dataset(df)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
# X shape: (53970, 10)
# Y shape: (53970,)
# exit(0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_test shape:", X_test.shape)
# 4. 构建多元线性回归模型
model = LinearRegression()
model.fit(X_train, Y_train)

# 5. 预测
Y_pred = model.predict(X_test)

# 6. 评估模型
mse = mean_squared_error(Y_test, Y_pred)
print(f"均方误差 (MSE): {mse}")



def sensitivity_analysis(model, X, feature_names, perturbation_range=np.linspace(-0.1, 0.1, 5)):
    """
    对模型进行敏感性分析。

    参数:
        model: 训练好的模型。
        X: 输入数据。
        feature_names: 特征名称列表。
        perturbation_range: 扰动范围（默认从-10%到+10%）。

    返回:
        sensitivity_results: 敏感性分析结果。
    """
    sensitivity_results = {}
    baseline_prediction = model.predict(X)  # 基准预测值

    for feature_idx, feature_name in enumerate(feature_names):
        sensitivity_results[feature_name] = []
        for perturbation in perturbation_range:
            X_perturbed = X.copy()
            # 对每个特征的时间序列数据进行扰动
            for year in range(10):
                X_perturbed[:, year] *= (1 + perturbation)  # 扰动第 year 年的特征值
            perturbed_prediction = model.predict(X_perturbed)
            sensitivity_results[feature_name].append(np.mean(perturbed_prediction - baseline_prediction))

    return sensitivity_results, perturbation_range


# 2. 选择关键特征进行敏感性分析
feature_names = ['资产总计', '资产负债率', '流动比率', '营业收入', '研发费用/营业总收入']
sensitivity_results, perturbation_range = sensitivity_analysis(model, X_test, feature_names)

# 3. 生成敏感性分析结果表格
sensitivity_df = pd.DataFrame(sensitivity_results, index=perturbation_range)
print("敏感性分析结果:")
print(sensitivity_df)
# exit(0)
# # 4. 可视化敏感性分析结果
# plt.figure(figsize=(10, 6))
# for feature in feature_names:
#     plt.plot(perturbation_range, sensitivity_results[feature], label=feature)
#
# plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # 基准线
# plt.title("敏感性分析")
# plt.xlabel("扰动比例")
# plt.ylabel("净利润变化")
# plt.legend()
# plt.grid(True)
# plt.show()

# 5. 保存敏感性分析结果
sensitivity_output_path = 'sensitivity_analysis.xlsx'
sensitivity_df.to_excel(sensitivity_output_path, index=True)
print(f"敏感性分析结果已保存到 {sensitivity_output_path}")





all_predicted_profits = []
future_years = 5
for i in tqdm(range(len(df))):
    # 获取第i家企业的过去10年数据
    past_data = df.iloc[i].values  # 第i家企业的数据

    # 构建未来5年的X值
    X_predict = []
    for year in range(10, 15):  # 未来5年（2024-2028）
        # 对每个特征，使用过去10年的数据进行回归，预测未来5年的值
        future_features = []
        for feature_idx in range(10):  # 假设有10个特征
            # 提取过去10年的特征值（跨年提取）
            past_feature_values = [past_data[year_idx + feature_idx * 10] for year_idx in range(10)]
            past_years = np.arange(2014, 2024).reshape(-1, 1)  # 过去10年的年份

            # 训练线性回归模型
            feature_model = LinearRegression()
            feature_model.fit(past_years, past_feature_values)

            # 预测未来5年的特征值
            future_feature_value = feature_model.predict([[2014 + year]])[0]
            future_features.append(future_feature_value)

        # 加入时间特征
        time_feature = 2014 + year  # 时间因素为年份
        future_features.append(time_feature)  # 将时间特征加入特征

        X_predict.append(future_features)

    # 使用多元线性回归模型预测未来5年的净利润
    X_predict = np.array(X_predict)
    predicted_profits = model.predict(X_predict)

    # 保存预测结果
    all_predicted_profits.append(predicted_profits)

# 8. 保存预测结果
predicted_df = pd.DataFrame(all_predicted_profits, columns=[f'Year_{i+1}' for i in range(future_years)])
predicted_output_file_path = 'predicted_profits_linear_regression.xlsx'
predicted_df.to_excel(predicted_output_file_path, index=False)
print(f"预测结果已保存到 {predicted_output_file_path}")

# # 2. 数据标准化
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)
# scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#
# # save scaled_df to excel
# scaled_output_file_path = 'scaled_data.xlsx'
# scaled_df.to_excel(scaled_output_file_path, index=False)
#
# print(type(scaled_df.iloc[0:0+10].values))
# print(scaled_df.iloc[0:0+10].values.shape)
# print(scaled_df.iloc[0+10:0+10+5].values) # 第10家到第15家企业的数据（含近十年净利润数据）
# print(scaled_df.iloc[0+10:0+10+5].values[:, -10:]) # 第10家到第15家企业的近十年净利润
#
#
# # 3. 准备训练数据
# def create_dataset(data):#, past_years=10, future_years=5):
#     # 9年的数据作为输入，1年的数据作为输出
#     X, Y = [], []
#     for i in range(len(data)):
#         past_data = data.iloc[i].values  # 第i家企业过去10年的数据
#         X.append(past_data[:110-1]) # 只取除 最后的2023年净利润 外的特征
#         future_data = data.iloc[i].values[-1:]  # 只取最后1列（净利润）
#         Y.append(future_data) #
#     return np.array(X), np.array(Y)
#
# past_years = 10
# future_years = 5
# X_train, Y_train = create_dataset(scaled_df)#, past_years, future_years)
#
# # 打印数据形状
# print("X_train shape:", X_train.shape)  # (样本数, 101)
# print("Y_train shape:", Y_train.shape)  # (样本数, 1)
#
# # 4. 构建LSTM模型
# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # 输入形状为 (109, 1)
# model.add(LSTM(50, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))  # 输出1个时间步的值
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), Y_train, epochs=100, batch_size=32, verbose=1)
# # exit(0)
#
#
# # 5. 自回归预测函数
# def autoregressive_predict(model, initial_input, future_steps):
#     predictions = []
#     current_input = initial_input  # 初始输入数据
#
#     for _ in tqdm(range(future_steps)):
#         # 预测下一个时间步的值
#         next_step = model.predict(current_input)
#         predictions.append(next_step[0, 0])  # 保存预测值
#
#         # 更新输入数据
#         current_input = np.concatenate([current_input[:, 1:, :], next_step.reshape(1, 1, -1)], axis=1)
#
#     return np.array(predictions)
#
# # 6. 对每一家企业预测未来5年的净利润
# future_years = 5
# all_predicted_profits = []
#
# for i in tqdm(range(len(scaled_df))):
#     # 获取第i家企业的数据
#     initial_input = scaled_df.iloc[i].values[:-1]  # 第i家企业的数据（除2023年净利润外）
#     initial_input = np.expand_dims(initial_input, axis=0).reshape(1, -1, 1)  # 增加批次维度并调整形状
#
#     # 自回归预测未来5年的净利润
#     predicted_profits = autoregressive_predict(model, initial_input, future_years)
#
#     # 反标准化
#     predicted_profits = scaler.inverse_transform(
#         np.concatenate([np.zeros((1, scaled_df.shape[1] - future_years)), predicted_profits.reshape(1, -1)], axis=1)
#     )[:, -future_years:]
#     # 保存预测结果
#     all_predicted_profits.append(predicted_profits.flatten())
#
# # 7. 保存预测结果
# predicted_df = pd.DataFrame(all_predicted_profits, columns=[f'Year_{i+1}' for i in range(future_years)])
# predicted_output_file_path = 'predicted_profits.xlsx'
# predicted_df.to_excel(predicted_output_file_path, index=False)
# print(f"预测结果已保存到 {predicted_output_file_path}")

# # 2. 数据标准化
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)
# scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#
# # save scaled_df to excel
# scaled_output_file_path = 'scaled_data.xlsx'
# scaled_df.to_excel(scaled_output_file_path, index=False)
#
# # 3. 准备训练数据
# def create_dataset(data, past_years=10, future_years=5):
#     X, Y = [], []
#     for i in range(len(data) - past_years - future_years + 1):
#         # 使用 loc 获取过去10年的数据
#         X.append(data.loc[data.index[i:i + past_years]].values)  # 过去10年的数据
#         # 使用 loc 获取未来5年的净利润（只提取净利润列）
#         Y.append(data.loc[data.index[i + past_years:i + past_years + future_years], '净利润-2014':'净利润-2023'].values[:, -1])  # 只取最后一列（净利润）
#     return np.array(X), np.array(Y)
#
# past_years = 10
# future_years = 5
# X, Y = create_dataset(scaled_df, past_years, future_years)
# # print(X[0])
# # print(Y[0])
# print(X.shape, Y.shape) # (5383, 10, 110) (5383, 5)
# # exit(0)
#
# # 4. 构建LSTM模型
# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(past_years, X.shape[2])))
# model.add(LSTM(50, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(future_years))  # 输出未来5年的净利润
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X, Y, epochs=100, batch_size=32, verbose=1)
#
# # 5. 预测未来净利润
# # 对每个企业的过去10年数据进行预测
# past_years = 10
# future_years = 5
# all_predicted_profits = []
# for i in range(len(scaled_df) - past_years + 1):
#     last_10_years = scaled_df.iloc[i:i + past_years].values
#     last_10_years = np.expand_dims(last_10_years, axis=0)  # 增加批次维度
#     predicted_profits = model.predict(last_10_years)
#     all_predicted_profits.append(predicted_profits[0])  # 保存预测结果
#
# # 将预测结果转换为数组
# all_predicted_profits = np.array(all_predicted_profits)
#
# # 反标准化
# all_predicted_profits = scaler.inverse_transform(
#     np.concatenate([np.zeros((all_predicted_profits.shape[0], scaled_df.shape[1] - future_years)), all_predicted_profits], axis=1)
# )[:, -future_years:]
#
# # 6. 保存预测结果
# predicted_df = pd.DataFrame(all_predicted_profits, columns=[f'Year_{i+1}' for i in range(future_years)])
# predicted_output_file_path = 'predicted_profits.xlsx'
# predicted_df.to_excel(predicted_output_file_path, index=False)
# print(f"预测结果已保存到 {predicted_output_file_path}")