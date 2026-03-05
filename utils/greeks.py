import copy
import numpy as np
from core.payoff import SnowballOption
from core.models import BaseModel
from core.engine import MonteCarloEngine

def calculate_delta_gamma(engine: MonteCarloEngine, option: SnowballOption, 
                          model: BaseModel, r: float, bump_ratio: float = 0.01) -> dict:
    """
    使用中心有限差分法 (Central Finite Difference) 配合公共随机数 (CRN) 计算 Delta 和 Gamma。
    
    参数:
    bump_ratio : 现价偏移比例，默认 1% (例如现价 100，则向上 bump 到 101，向下 bump 到 99)
    """
    # 获取当前基础状态
    S0_base = model.S0
    dS = S0_base * bump_ratio
    
    # 为了防止修改原对象，我们使用深拷贝
    # 1. Base (基础价格)
    pv_base = engine.calculate_pv(option, model, r)['price']
    
    # 2. Bump Up (向上偏移)
    model_up = copy.deepcopy(model)
    model_up.S0 = S0_base + dS
    option_up = copy.deepcopy(option)
    option_up.initial_price = option.initial_price # 注意：期初价格(用于算敲入亏损)不变
    pv_up = engine.calculate_pv(option_up, model_up, r)['price']
    
    # 3. Bump Down (向下偏移)
    model_down = copy.deepcopy(model)
    model_down.S0 = S0_base - dS
    option_down = copy.deepcopy(option)
    option_down.initial_price = option.initial_price
    pv_down = engine.calculate_pv(option_down, model_down, r)['price']
    
    # 4. 计算 Greeks
    # 注意：计算出的 Delta 是每 1 块钱名义本金对应的标的资产暴露
    delta = (pv_up - pv_down) / (2 * dS)
    gamma = (pv_up - 2 * pv_base + pv_down) / (dS ** 2)
    
    return {
        "base_price": pv_base,
        "delta": delta,
        "gamma": gamma
    }