"""
参数优化模块

提供策略参数的网格搜索和随机搜索功能。
"""

import logging
import random
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from openquant.engine.backtest_engine import BacktestEngine
from openquant.core.models import MarketType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """参数搜索空间定义"""
    name: str
    param_type: str  # "int", "float", "choice"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    
    def __post_init__(self):
        if self.param_type not in ["int", "float", "choice"]:
            raise ValueError(f"param_type must be 'int', 'float', or 'choice', got {self.param_type}")
        
        if self.param_type == "choice" and self.choices is None:
            raise ValueError("choices must be provided for 'choice' type")
        
        if self.param_type in ["int", "float"]:
            if self.low is None or self.high is None:
                raise ValueError(f"low and high must be provided for '{self.param_type}' type")
            
            if self.step is None:
                self.step = 1.0 if self.param_type == "int" else 0.01


@dataclass
class OptimizationResult:
    """单次优化结果"""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    target_value: float


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(
        self,
        strategy_class: type,
        data_feeds: List[Tuple[str, pd.DataFrame, MarketType]],
        initial_capital: float,
        commission_rate: float,
        slippage_rate: float,
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_symbol: Optional[str] = None
    ):
        """
        初始化参数优化器
        
        Args:
            strategy_class: 策略类
            data_feeds: 数据源列表，每个元素为 (symbol, data, market_type)
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            benchmark_data: 基准数据（可选）
            benchmark_symbol: 基准代码（可选）
        """
        self.strategy_class = strategy_class
        self.data_feeds = data_feeds
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.benchmark_data = benchmark_data
        self.benchmark_symbol = benchmark_symbol
        
        self.parameter_spaces: List[ParameterSpace] = []
        self.results: List[OptimizationResult] = []
    
    def add_parameter(
        self,
        name: str,
        param_type: str,
        low: Optional[float] = None,
        high: Optional[float] = None,
        step: Optional[float] = None,
        choices: Optional[List[Any]] = None
    ):
        """
        添加参数到搜索空间
        
        Args:
            name: 参数名
            param_type: 参数类型 ("int", "float", "choice")
            low: 最小值（数值类型必需）
            high: 最大值（数值类型必需）
            step: 步长（可选）
            choices: 可选值列表（choice 类型必需）
        """
        param_space = ParameterSpace(
            name=name,
            param_type=param_type,
            low=low,
            high=high,
            step=step,
            choices=choices
        )
        self.parameter_spaces.append(param_space)
        logger.info(f"Added parameter: {name} ({param_type})")
    
    def grid_search(self, target_metric: str, maximize: bool = True) -> List[OptimizationResult]:
        """
        网格搜索
        
        Args:
            target_metric: 目标指标名称
            maximize: 是否最大化目标指标
            
        Returns:
            优化结果列表（按目标值排序）
        """
        logger.info("Starting grid search...")
        self.results = []
        
        combinations = self._generate_grid_combinations()
        total_combinations = len(combinations)
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        for idx, params in enumerate(combinations, 1):
            logger.info(f"Running combination {idx}/{total_combinations}: {params}")
            result = self._run_single_backtest(params, target_metric)
            
            if result:
                self.results.append(result)
        
        # 排序结果
        reverse = maximize
        self.results.sort(key=lambda x: x.target_value, reverse=reverse)
        
        logger.info(f"Grid search completed. Total results: {len(self.results)}")
        return self.results
    
    def random_search(
        self,
        target_metric: str,
        num_trials: int,
        maximize: bool = True
    ) -> List[OptimizationResult]:
        """
        随机搜索
        
        Args:
            target_metric: 目标指标名称
            num_trials: 随机试验次数
            maximize: 是否最大化目标指标
            
        Returns:
            优化结果列表（按目标值排序）
        """
        logger.info(f"Starting random search with {num_trials} trials...")
        self.results = []
        
        for idx in range(1, num_trials + 1):
            params = self._generate_random_params()
            logger.info(f"Running trial {idx}/{num_trials}: {params}")
            result = self._run_single_backtest(params, target_metric)
            
            if result:
                self.results.append(result)
        
        # 排序结果
        reverse = maximize
        self.results.sort(key=lambda x: x.target_value, reverse=reverse)
        
        logger.info(f"Random search completed. Total results: {len(self.results)}")
        return self.results
    
    def get_results(self) -> List[OptimizationResult]:
        """获取所有优化结果"""
        return self.results
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取最优参数"""
        if not self.results:
            return {}
        return self.results[0].params
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """将结果转换为 DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for result in self.results:
            row = {**result.params, **result.metrics}
            row['target_value'] = result.target_value
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _run_single_backtest(self, params: Dict[str, Any], target_metric: str) -> Optional[OptimizationResult]:
        """
        运行单次回测
        
        Args:
            params: 策略参数
            target_metric: 目标指标名称
            
        Returns:
            优化结果
        """
        try:
            # 创建回测引擎
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                slippage_rate=self.slippage_rate,
                storage=None,
                stop_loss_config=None,
                risk_manager=None
            )
            
            # 创建策略实例
            strategy = self.strategy_class(**params)
            engine.set_strategy(strategy)
            
            # 添加数据
            for symbol, data, market_type in self.data_feeds:
                engine.add_data(symbol, data, market_type)
            
            # 设置基准
            if self.benchmark_data is not None and self.benchmark_symbol is not None:
                engine.set_benchmark(self.benchmark_symbol, self.benchmark_data)
            
            # 运行回测
            portfolio = engine.run()
            
            # 获取结果
            metrics = engine.get_results()
            
            # 获取目标指标值
            target_value = metrics.get(target_metric, 0.0)
            
            return OptimizationResult(
                params=params.copy(),
                metrics=metrics,
                target_value=target_value
            )
            
        except Exception as e:
            logger.error(f"Error running backtest with params {params}: {str(e)}")
            return None
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """生成网格参数组合"""
        value_lists = []
        
        for param_space in self.parameter_spaces:
            if param_space.param_type == "int":
                values = list(range(
                    int(param_space.low),
                    int(param_space.high) + 1,
                    int(param_space.step)
                ))
            elif param_space.param_type == "float":
                values = []
                current = param_space.low
                while current <= param_space.high + 1e-9:
                    values.append(round(current, 10))
                    current += param_space.step
            else:  # choice
                values = param_space.choices
            
            value_lists.append(values)
        
        # 生成所有组合
        combinations = list(itertools.product(*value_lists))
        
        # 转换为字典列表
        param_dicts = []
        for combo in combinations:
            param_dict = {}
            for i, param_space in enumerate(self.parameter_spaces):
                param_dict[param_space.name] = combo[i]
            param_dicts.append(param_dict)
        
        return param_dicts
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """生成随机参数"""
        params = {}
        
        for param_space in self.parameter_spaces:
            if param_space.param_type == "int":
                num_steps = int((param_space.high - param_space.low) / param_space.step)
                random_step = random.randint(0, num_steps)
                value = int(param_space.low + random_step * param_space.step)
            elif param_space.param_type == "float":
                num_steps = int((param_space.high - param_space.low) / param_space.step)
                random_step = random.randint(0, num_steps)
                value = param_space.low + random_step * param_space.step
                value = round(value, 10)
            else:  # choice
                value = random.choice(param_space.choices)
            
            params[param_space.name] = value
        
        return params
