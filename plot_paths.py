import matplotlib.pyplot as plt
from core.sde import generate_gbm_paths

def main():
    # 模拟市场环境参数
    S0 = 100.0      # 标的资产初始价格
    r = 0.03        # 无风险利率 3%
    q = 0.0         # 股息率 0%
    vol = 0.20      # 年化波动率 20%
    T = 1.0         # 到期时间 1年
    n_steps = 252   # 1年按 252 个交易日离散化
    n_paths = 100   # 画图只用 100 条路径（实际定价通常 10 万条起步）

    print("正在生成 GBM 路径...")
    # 调用我们刚刚写的纯 NumPy 向量化引擎
    paths = generate_gbm_paths(S0, r, q, vol, T, n_steps, n_paths, seed=42)
    print(f"路径生成完毕！矩阵形状: {paths.shape}")

    # 使用 Matplotlib 进行专业的可视化
    plt.figure(figsize=(10, 6))
    plt.plot(paths, lw=1.0, alpha=0.8) # lw是线宽，alpha是透明度，让图看起来更高级
    
    plt.title('Geometric Brownian Motion - Monte Carlo Simulated Paths', fontsize=14)
    plt.xlabel('Time Steps (Trading Days)', fontsize=12)
    plt.ylabel('Underlying Asset Price', fontsize=12)
    
    # 画一条初始价格的水平参考线
    plt.axhline(y=S0, color='r', linestyle='--', label='Initial Spot Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("正在生成图表...")
    plt.show()

if __name__ == "__main__":
    main()