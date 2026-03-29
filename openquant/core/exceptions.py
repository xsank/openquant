"""自定义异常类"""


class OpenQuantError(Exception):
    """OpenQuant 基础异常"""


class DataSourceError(OpenQuantError):
    """数据源相关异常"""


class StorageError(OpenQuantError):
    """存储相关异常"""


class StrategyError(OpenQuantError):
    """策略相关异常"""


class InsufficientFundsError(OpenQuantError):
    """资金不足异常"""


class InsufficientPositionError(OpenQuantError):
    """持仓不足异常"""


class InvalidOrderError(OpenQuantError):
    """无效订单异常"""
