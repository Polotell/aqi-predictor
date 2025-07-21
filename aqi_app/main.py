# 文件名: main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # 新增: 处理跨域请求
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer # 确保导入

# 创建 FastAPI 应用实例
app = FastAPI(title="空气质量指数预测API")

# 配置 CORS，允许前端访问你的后端API
# 生产环境中，你应该把"*"替换成你的前端域名，例如 "https://yourfrontend.onrender.com"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有来源访问，本地测试方便。部署时建议改为你的前端域名。
    allow_credentials=True,
    allow_methods=["*"], # 允许所有HTTP方法 (GET, POST等)
    allow_headers=["*"], # 允许所有请求头
)

# 全局变量来存储模型和imputer，确保只加载一次
model = None
imputer = None
features = ['co', 'nox', 'no2', 'o3', 'pm10', 'pm25'] # 保持与训练时一致

@app.on_event("startup")
async def load_model_and_imputer():
    """应用启动时加载模型和imputer"""
    global model, imputer
    try:
        # 确保这两个文件与你的 main.py 在同一个文件夹下
        model = joblib.load('aqi_mlp_model.pkl')
        imputer = joblib.load('aqi_imputer.pkl')
        print("模型和Imputer加载成功！")
    except FileNotFoundError:
        print("错误: 模型文件或Imputer文件未找到。请确保 'aqi_mlp_model.pkl' 和 'aqi_imputer.pkl' 存在。")
        # 在实际部署中，这里可能需要更优雅的错误处理，但对于简单教程，直接抛出异常。
        raise HTTPException(status_code=500, detail="模型或Imputer加载失败，文件不存在。")
    except Exception as e:
        print(f"加载模型或Imputer时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"模型或Imputer加载失败: {e}")

# 定义输入数据模型，FastAPI 会自动验证输入数据
class AQIInput(BaseModel):
    co: float
    nox: float
    no2: float
    o3: float
    pm10: float
    pm25: float

@app.post("/predict_aqi/")
async def predict_aqi(data: AQIInput):
    """
    接收空气污染物数据并预测AQI。
    """
    if model is None or imputer is None:
        raise HTTPException(status_code=500, detail="模型或Imputer未加载。请检查服务器状态。")

    # 将输入数据转换为Pandas DataFrame
    # 注意: FastAPI 的 data.dict() 方法在 Pydantic v2 中已更名为 data.model_dump()
    # 为了兼容性，这里使用 data.dict()，但在新版本Pydantic中，可能需要调整。
    input_data_dict = data.dict()
    input_df = pd.DataFrame([input_data_dict])

    # 确保输入数据的列顺序与训练模型时使用的特征顺序一致
    # 这一步非常重要，否则模型会得到错误的预测结果
    input_df = input_df[features]

    # 使用之前加载的imputer处理输入数据中的缺失值
    # imputer.transform() 会返回一个 NumPy 数组
    imputed_data = imputer.transform(input_df)

    # 使用模型进行预测
    # .predict() 返回一个数组，我们取第一个元素作为预测值
    prediction = model.predict(imputed_data)[0]

    # 返回预测结果，转换为 float 类型以确保JSON序列化正确
    return {"AQI_calculated": float(prediction)}

# 这是为了在本地运行 FastAPI 应用而添加的代码
# 在部署到Render时，Render会使用Procfile来启动应用，所以这部分代码在部署时不会直接执行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)