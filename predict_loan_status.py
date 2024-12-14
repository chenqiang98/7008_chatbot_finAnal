# 定义输入格式
import joblib
import pandas as pd

out_dict = {
    0: "Charged Off",
    1: "Current",
    2: "Default",
    3: "Does not meet the credit policy. Status:Charged Off",
    4: "Not meet the credit policy. Status:Fully Paid",
    5: "Fully Paid",
    6: "In Grace Period",
    7: "Issued",
    8: "Late (16-30 days)",
    9: "Late (31-120 days)"
}

status_dict = { 0: "In high risk",
                1: "Normal",
                2: "In high risk",
                3: "Not recommended for ratification",
                4: "Normal",
                5: "Normal",
                6: "Normal",
                7: "Normal",
                8: "Risk Warning",
                9: "Risk Warning"    
               }

def input_info(list):
    # list = [grade, emp_title, emp_length, home_ownership, annual_inc, verification_status, dti, delinq_2yrs, tot_cur_bal, total_rev_hi_lim]
    res = pd.DataFrame([list], columns=['grade', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'dti', 'delinq_2yrs', 'tot_cur_bal', 'total_rev_hi_lim'])
    le = joblib.load("models/LabelEncoder.pkl")
    # print(res.columns)
    for col in res.columns:
        if res[col].dtype == 'object':
                res[col] = le.fit_transform(res[col].astype(str))  # 将分类特征编码为数值
                # print(res)
        elif res[col].dtype == 'datetime64[ns]':
            res[col] = (res[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  # 转换为自 1970 年以来的秒数
        elif not pd.api.types.is_numeric_dtype(res[col]):
            res[col] = pd.to_numeric(res[col], errors='coerce')  # 将其他类型转换为数值，无法转换的填为 NaN
    return res

def model_pred_status(data_item):
    loaded_model = joblib.load("models/DecisionTree.pkl")
    ans = loaded_model.predict(data_item)
    return status_dict[ans[0]]

# 使用例
# tmp = input_info("C", "KPMG", "7 years", "RENT", "59000", "Verified", "31.04", "0", "37476", "35800")
# loaded_model = joblib.load("DecisionTree.pkl")
# ans = model.predict(tmp)
# print(ans)

if __name__ == '__main__':
    res = model_pred_status(input_info(["C", "KPMG", "7 years", "RENT", "59000", "Verified", "31.04", "0", "37476", "35800"]))
    print(out_dict[res[0]])