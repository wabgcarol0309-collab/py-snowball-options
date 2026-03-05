import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.payoff import SnowballOption
from core.engine import MonteCarloEngine
from core.models import GBM
from utils.greeks import calculate_delta_gamma

def main():
    print("正在计算雪球期权 Gamma 3D 曲面 (耗时可能较长，请耐心等待)...\n")
    
    # 设定固定的环境参数
    initial_spot = 100.0
    r = 0.03
    q = 0.0
    vol = 0.15
    
    # 合约参数：103敲出，80敲入
    ko_barrier = 103.0
    ki_barrier = 80.0
    coupon = 0.20
    notional = 1.0  # 画图时为了刻度清晰，名义本金设为 1
    
    # 引擎设定：为了画图快一点，路径数稍微调小到 20000
    engine = MonteCarloEngine(n_paths=20000, n_steps=252, seed=42)
    
    # 构建网格 (Meshgrid)
    # 标的价格区间：从深度敲入区 (70) 到敲出区 (105)
    spot_range = np.linspace(70, 105, 20)
    # 剩余期限区间：从 1 年到 0.05 年 (快到期)
    time_range = np.linspace(1.0, 0.05, 10)
    
    S_mesh, T_mesh = np.meshgrid(spot_range, time_range)
    Gamma_mesh = np.zeros_like(S_mesh)
    
    # 嵌套循环计算网格上每一个点的 Gamma
    for i in range(len(time_range)):
        for j in range(len(spot_range)):
            current_T = T_mesh[i, j]
            current_S = S_mesh[i, j]
            
            # 动态调整观察日 (简化处理，按比例缩减观察日)
            n_steps = max(int(252 * current_T), 10)
            engine.n_steps = n_steps
            obs_indices = np.arange(21, n_steps+1, 21) if n_steps > 21 else np.array([n_steps])
            
            option = SnowballOption(
                initial_price=initial_spot, ko_barrier=ko_barrier, ki_barrier=ki_barrier,
                coupon_rate=coupon, ko_obs_indices=obs_indices, T=current_T, notional=notional
            )
            model = GBM(S0=current_S, r=r, q=q, vol=vol)
            
            # 核心调用：计算 Greeks
            greeks = calculate_delta_gamma(engine, option, model, r, bump_ratio=0.01)
            Gamma_mesh[i, j] = greeks['gamma']
            
        print(f"进度: 剩余期限 T = {current_T:.2f} 年已计算完成.")

    # 绘制高级 3D 曲面图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(S_mesh, T_mesh, Gamma_mesh, cmap='coolwarm', 
                           edgecolor='none', alpha=0.8)
    
    ax.set_title('Snowball Option Gamma Surface (Pin Risk at Knock-in Barrier)', fontsize=14)
    ax.set_xlabel('Underlying Price (Spot)', fontsize=12)
    ax.set_ylabel('Time to Maturity (Years)', fontsize=12)
    ax.set_zlabel('Gamma', fontsize=12)
    
    # 标记敲入线
    ax.plot([ki_barrier]*10, time_range, np.zeros_like(time_range), 
            color='red', linestyle='dashed', linewidth=2, label='Knock-in Barrier (80)')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()