import numpy as np

def generate_gbm_paths(S0: float, r: float, q: float, vol: float, 
                       T: float, n_steps: int, n_paths: int, 
                       seed: int = None) -> np.ndarray:
    """
    使用纯 NumPy 向量化生成几何布朗运动 (GBM) 价格路径。
    
    参数:
    S0      : float, 初始资产价格
    r       : float, 无风险连续复利利率
    q       : float, 连续股息率
    vol     : float, 年化波动率
    T       : float, 到期时间（以年为单位）
    n_steps : int, 模拟的时间步数
    n_paths : int, 模拟的路径数量
    seed    : int, 随机数种子（用于保证结果可复现）
    
    返回:
    np.ndarray: 形状为 (n_steps + 1, n_paths) 的二维数组，包含所有模拟路径。
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / n_steps
    
    # 1. 一次性生成所有所需的标准正态分布随机数矩阵
    Z = np.random.standard_normal((n_steps, n_paths))
    
    # 2. 预分配路径矩阵内存，第一行初始化为 S0
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0
    
    # 3. 计算常数漂移项 (Drift) 和扩散项系数 (Diffusion)
    drift = (r - q - 0.5 * vol ** 2) * dt
    diffusion = vol * np.sqrt(dt)
    
    # 4. 极致向量化：计算对数收益率的增量，并沿时间轴(axis=0)累加
    # 这样彻底避免了 Python 层面的时间步 for 循环
    log_returns = drift + diffusion * Z
    paths[1:] = S0 * np.exp(np.cumsum(log_returns, axis=0))
    
    return paths