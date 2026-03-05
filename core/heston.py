import numpy as np

def generate_heston_paths(S0: float, v0: float, r: float, q: float, 
                          kappa: float, theta: float, sigma_v: float, rho: float,
                          T: float, n_steps: int, n_paths: int, 
                          seed: int = None) -> np.ndarray:
    """
    使用 Euler-Maruyama 全截断方案 (Full Truncation) 生成 Heston 模型价格路径。
    
    参数:
    S0      : 初始标的价格
    v0      : 初始方差 (注意是方差不是波动率)
    kappa   : 均值回归速度
    theta   : 长期平均方差
    sigma_v : 波动率的波动率 (vol-of-vol)
    rho     : 价格与方差的布朗运动相关系数
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / n_steps
    
    # 1. 生成两组独立的标准正态随机数矩阵
    Z1 = np.random.standard_normal((n_steps, n_paths))
    Z2 = np.random.standard_normal((n_steps, n_paths))
    
    # 2. 通过 Cholesky 分解构造相关的布朗运动增量
    # Z_S 用于价格路径，Z_v 用于方差路径
    Z_S = Z1
    Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # 3. 预分配内存
    S_paths = np.zeros((n_steps + 1, n_paths))
    v_paths = np.zeros((n_steps + 1, n_paths))
    
    S_paths[0] = S0
    v_paths[0] = v0
    
    # 4. 时间步迭代 (Heston 模型由于方差依赖前一步状态，无法像 GBM 那样完全消除时间轴循环)
    # 但我们仍然在路径维度 (n_paths) 上保持了纯向量化
    for t in range(n_steps):
        # 当前方差
        v_t = v_paths[t]
        
        # 全截断处理 (Full Truncation)：将负方差视为 0 参与后续计算
        v_t_plus = np.maximum(v_t, 0.0)
        
        # 计算下一步的方差 (Euler-Maruyama 离散化)
        # dv_t = kappa * (theta - v_t_plus) * dt + sigma_v * sqrt(v_t_plus) * sqrt(dt) * Z_v
        v_paths[t+1] = v_t + kappa * (theta - v_t_plus) * dt + \
                       sigma_v * np.sqrt(v_t_plus) * np.sqrt(dt) * Z_v[t]
        
        # 计算下一步的对数价格并转回价格
        # 使用对数价格离散化更稳定：d(lnS_t) = (r - q - 0.5 * v_t_plus) * dt + sqrt(v_t_plus) * dW_t^S
        log_S_next = np.log(S_paths[t]) + (r - q - 0.5 * v_t_plus) * dt + \
                     np.sqrt(v_t_plus) * np.sqrt(dt) * Z_S[t]
                     
        S_paths[t+1] = np.exp(log_S_next)
        
    return S_paths # 如果需要，也可以把 v_paths 一起返回用于波动率分析