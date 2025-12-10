"""
Базовый класс для всех моделей прогнозирования временных рядов.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, Optional


class BaseTimeSeriesModel(ABC):
    """
    Абстрактный базовый класс для моделей временных рядов.
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Название модели
        """
        self.name = name
        self.fitted_model = None
        self.params = {}
        self.initial_params = None  # For federated learning: start from these params
        self.train_data = None
        self.transformation_params = {}
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'BaseTimeSeriesModel':
        """
        Обучение модели на тренировочных данных.
        
        Args:
            train_data: DataFrame с данными для обучения
            **kwargs: Дополнительные параметры модели
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """
        Прогноз на N шагов вперед.
        
        Args:
            steps: Количество шагов прогноза
            
        Returns:
            Массив с прогнозными значениями
        """
        pass
    
    def normalize(self, data: pd.Series, eps: float = 1e-6) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Нормализация данных в диапазон [0, 1].
        
        Args:
            data: Исходные данные
            eps: Малая константа для избежания деления на ноль
            
        Returns:
            Нормализованные данные и параметры трансформации
        """
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + eps) + eps
        
        params = {'min': min_val, 'max': max_val, 'eps': eps}
        return normalized, params
    
    def denormalize(self, data: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Обратная нормализация данных.
        
        Args:
            data: Нормализованные данные
            params: Параметры трансформации из normalize()
            
        Returns:
            Денормализованные данные
        """
        return (data - params['eps']) * (params['max'] - params['min']) + params['min']
    
    def boxcox_transform(self, data: pd.Series) -> Tuple[pd.Series, float]:
        """
        Box-Cox преобразование данных.
        
        Args:
            data: Исходные данные
            
        Returns:
            Преобразованные данные и параметр lambda
        """
        transformed, lmbda = stats.boxcox(data)
        return pd.Series(transformed, index=data.index), lmbda
    
    def inverse_boxcox(self, data: np.ndarray, lmbda: float) -> np.ndarray:
        """
        Обратное Box-Cox преобразование.
        
        Args:
            data: Преобразованные данные
            lmbda: Параметр lambda из boxcox_transform()
            
        Returns:
            Исходные данные
        """
        if lmbda == 0:
            return np.exp(data)
        else:
            return np.exp(np.log(lmbda * data + 1) / lmbda)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Получение параметров модели.
        
        Returns:
            Словарь с параметрами
        """
        return self.params.copy()
    
    def set_params(self, params: Dict[str, Any]) -> 'BaseTimeSeriesModel':
        """
        Установка параметров модели.
        
        Args:
            params: Словарь с параметрами
            
        Returns:
            self
        """
        self.params.update(params)
        return self
    
    def set_initial_params(self, params: Dict[str, Any]) -> 'BaseTimeSeriesModel':
        """
        Установка начальных параметров для federated learning.
        Модель будет использовать эти параметры как начальную точку при fit().
        
        Args:
            params: Словарь с параметрами для инициализации
            
        Returns:
            self
        """
        self.initial_params = params.copy() if params else None
        return self
    
    def _build_start_params(self, params_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Build start_params array from initial_params dictionary for statsmodels.
        This is used for parameter initialization in federated learning.
        
        The key insight: statsmodels expects start_params in a SPECIFIC ORDER
        that matches the model's param_names. We try to match by name when possible.
        
        Args:
            params_dict: Dictionary of parameters (typically from get_params())
            
        Returns:
            Array of parameter values, or None if unable to build
        """
        if not params_dict:
            return None
        
        try:
            # If we have a fitted model, use its param structure
            if self.fitted_model is not None:
                param_names = self.fitted_model.param_names
                values = []
                
                for param_name in param_names:
                    # Try exact match
                    if param_name in params_dict:
                        val = params_dict[param_name]
                        if isinstance(val, (int, float, np.number)):
                            values.append(float(val))
                        else:
                            # Skip non-numeric params
                            values.append(1.0)
                    else:
                        # Try partial match (e.g., 'ar.L1' might match 'ar.L1' or 'ar_L1')
                        found = False
                        for key in params_dict:
                            if key.replace('_', '.') == param_name or key.replace('.', '_') == param_name:
                                val = params_dict[key]
                                if isinstance(val, (int, float, np.number)):
                                    values.append(float(val))
                                found = True
                                break
                        if not found:
                            # Use default: 0 for AR/MA, 1.0 for sigma
                            if 'sigma' in param_name:
                                values.append(1.0)
                            else:
                                values.append(0.1)
                
                if values:
                    return np.array(values)
            
            return None
        except Exception:
            return None
    
    def predict_with_params(self, params: Dict[str, Any], steps: int) -> np.ndarray:
        """
        Predict using specific parameters WITHOUT retraining the model.
        Used in federated learning to evaluate effect of aggregated parameters.
        
        This is a fallback implementation that trains once and returns prediction.
        Subclasses should override if they want true parameter-swapping without retraining.
        
        Args:
            params: Parameter dictionary to use
            steps: Number of steps to predict
            
        Returns:
            Array with predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call fit() before predict_with_params().")
        
        # For now: just use existing fitted model
        # In future: could do parameter swapping at state-space level
        return self.predict(steps=steps)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
