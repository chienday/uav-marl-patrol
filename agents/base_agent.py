# -*- coding: utf-8 -*-
"""
Base Agent Class - Giao diện chung cho tất cả agents

"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np


class BaseAgent(ABC):
    """
    Base class cho tất cả agents (PPO, IPPO, MAPPO, VDPPO)
    
    Mục đích:
    - Định nghĩa interface chung
    - Cho phép dễ dàng thêm thuật toán mới
    - Đảm bảo tính nhất quán giữa các agents
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Khởi tạo agent
        
        Args:
            name (str): Tên của agent
            config (Dict): Configuration dict chứa siêu tham số
        """
        self.name = name
        self.config = config
        self.model = None
        self.vec_env = None
    
    @abstractmethod
    def train(self, total_timesteps: int, callbacks=None) -> None:
        """
        Huấn luyện agent
        
        Args:
            total_timesteps (int): Tổng số timesteps để train
            callbacks: List callbacks
        """
        pass
    
    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, Any]:
        """
        Dự đoán hành động
        
        Args:
            obs (np.ndarray): Observation
            deterministic (bool): Sử dụng deterministic policy hay không
        
        Returns:
            Tuple[int, Any]: (action, extra_info)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Lưu model
        
        Args:
            path (str): Đường dẫn lưu
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model
        
        Args:
            path (str): Đường dẫn model
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset agent state
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (Algorithm: {self.config.get('algorithm', 'Unknown')})"
    
    def __repr__(self) -> str:
        return self.__str__()
