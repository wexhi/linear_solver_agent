from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 🔥 引入 CORS 处理
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from langchain_core.messages import AIMessage, ToolMessage
from app.graph import graph

app = FastAPI()

# ✅ 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名访问，可以改为 ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, OPTIONS, DELETE, etc.)
    allow_headers=["*"],  # 允许所有请求头
)

# 定义请求的数据格式
class LinearEquationRequest(BaseModel):
    equations: List[str]
    method: str
    tolerance: float
    max_iterations: int

# `solution` 变为可选，新增 `steps` 存储详细推理过程
class LinearEquationResponse(BaseModel):
    solution: Optional[str] = None  # 允许 `solution` 缺失
    steps: List[str]  # 详细的推理过程

@app.post("/solve", response_model=LinearEquationResponse)
async def solve_linear_equations(request: LinearEquationRequest):
    """接收用户输入的方程组，并调用 LangChain Agent 计算，返回完整推理过程"""
    prompt = (
        f"Solve the following system of linear equations using the {request.method} method:\n"
        + "\n".join(request.equations)
        + f"\nwith an error tolerance of {request.tolerance} and a maximum iteration limit of {request.max_iterations}."
    )

    try:
        # 调用 LangChain Agent
        response = await graph.ainvoke({"messages": [prompt]})

        # 提取 AI 生成的所有消息
        messages = response.get("messages", [])

        # 解析对话内容
        steps = [msg.content for msg in messages if isinstance(msg, (AIMessage, ToolMessage))]

        # 取最后一个作为最终解
        final_solution = steps[-1] if steps else None

        return {"solution": final_solution, "steps": steps}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
