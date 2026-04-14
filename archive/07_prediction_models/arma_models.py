"""
Реализация AR, MA, ARMA и ARMAX моделей для прогнозирования временных рядов.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Tuple, Dict, Any
import warnings

from base_model import BaseTimeSeriesModel

warnings.filterwarnings('ignore')


class ARModel(BaseTimeSeriesModel):
    """
    Авторегрессионная модель (AR).
    """
    
    def __init__(self, name: str = "AR", p: int = 1):
        """
        Args:
            name: Название модели
            p: Порядок авторегрессии
        """
        super().__init__(name)
        self.p = p
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt', 
            use_transform: bool = True) -> 'ARModel':
        """
        Обучение AR модели.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            use_transform: Использовать ли трансформацию данных
            
        Returns:
            self
        """
        self.train_data = train_data.copy()
        data = train_data[target_col].copy()
        
        if use_transform:
            # Нормализация
            norm_data, norm_params = self.normalize(data)
            self.transformation_params['normalize'] = norm_params
            
            # Box-Cox
            transformed_data, lmbda = self.boxcox_transform(norm_data)
            self.transformation_params['boxcox_lambda'] = lmbda
        else:
            transformed_data = data
            
        # Обучение AR модели
        model = AutoReg(transformed_data, lags=self.p)
        self.fitted_model = model.fit()
        
        # Сохранение параметров
        self.params = dict(self.fitted_model.params)
        
        return self
    
    def predict(self, steps: int, use_transform: bool = True) -> np.ndarray:
        """
        Прогноз на N шагов вперед.
        
        Args:
            steps: Количество шагов прогноза
            use_transform: Использовались ли трансформации при обучении
            
        Returns:
            Массив прогнозных значений
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        
        # Получение прогноза
        forecast = self.fitted_model.forecast(steps=steps)
        
        if use_transform:
            # Обратная Box-Cox трансформация (guard missing params)
            lmbda = self.transformation_params.get('boxcox_lambda') if isinstance(self.transformation_params, dict) else None
            if lmbda is not None:
                try:
                    forecast = self.inverse_boxcox(forecast, lmbda)
                except Exception:
                    pass

            # Обратная нормализация (guard)
            norm_params = self.transformation_params.get('normalize') if isinstance(self.transformation_params, dict) else None
            if norm_params:
                try:
                    forecast = self.denormalize(forecast, norm_params)
                except Exception:
                    pass
        
        return forecast


class MAModel(BaseTimeSeriesModel):
    """
    Модель скользящего среднего (MA).
    """
    
    def __init__(self, name: str = "MA", q: int = 1):
        """
        Args:
            name: Название модели
            q: Порядок скользящего среднего
        """
        super().__init__(name)
        self.q = q
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            use_transform: bool = True) -> 'MAModel':
        """
        Обучение MA модели.
        """
        self.train_data = train_data.copy()
        data = train_data[target_col].copy()
        
        if use_transform:
            norm_data, norm_params = self.normalize(data)
            self.transformation_params['normalize'] = norm_params
            transformed_data, lmbda = self.boxcox_transform(norm_data)
            self.transformation_params['boxcox_lambda'] = lmbda
        else:
            transformed_data = data
            
        # MA модель = ARIMA(0, 0, q)
        model = ARIMA(transformed_data, order=(0, 0, self.q))
        self.fitted_model = model.fit()
        self.params = dict(self.fitted_model.params)
        
        return self
    
    def predict(self, steps: int, use_transform: bool = True) -> np.ndarray:
        """Прогноз на N шагов вперед."""
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        
        forecast = self.fitted_model.forecast(steps=steps)
        
        if use_transform:
            lmbda = self.transformation_params.get('boxcox_lambda') if isinstance(self.transformation_params, dict) else None
            if lmbda is not None:
                try:
                    forecast = self.inverse_boxcox(forecast, lmbda)
                except Exception:
                    pass
            norm_params = self.transformation_params.get('normalize') if isinstance(self.transformation_params, dict) else None
            if norm_params:
                try:
                    forecast = self.denormalize(forecast, norm_params)
                except Exception:
                    pass
        
        return forecast


class ARMAModel(BaseTimeSeriesModel):
    """
    Авторегрессионная модель скользящего среднего (ARMA).
    """
    
    def __init__(self, name: str = "ARMA", p: int = 1, q: int = 1):
        """
        Args:
            name: Название модели
            p: Порядок авторегрессии
            q: Порядок скользящего среднего
        """
        super().__init__(name)
        self.p = p
        self.q = q
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            use_transform: bool = True) -> 'ARMAModel':
        """
        Обучение ARMA модели.
        """
        self.train_data = train_data.copy()
        data = train_data[target_col].copy()
        
        if use_transform:
            norm_data, norm_params = self.normalize(data)
            self.transformation_params['normalize'] = norm_params
            transformed_data, lmbda = self.boxcox_transform(norm_data)
            self.transformation_params['boxcox_lambda'] = lmbda
        else:
            transformed_data = data
            
        # ARMA модель = ARIMA(p, 0, q)
        model = ARIMA(transformed_data, order=(self.p, 0, self.q))
        self.fitted_model = model.fit()
        self.params = dict(self.fitted_model.params)
        
        return self
    
    def predict(self, steps: int, use_transform: bool = True) -> np.ndarray:
        """Прогноз на N шагов вперед."""
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        
        forecast = self.fitted_model.forecast(steps=steps)
        
        if use_transform:
            lmbda = self.transformation_params.get('boxcox_lambda') if isinstance(self.transformation_params, dict) else None
            if lmbda is not None:
                try:
                    forecast = self.inverse_boxcox(forecast, lmbda)
                except Exception:
                    pass
            norm_params = self.transformation_params.get('normalize') if isinstance(self.transformation_params, dict) else None
            if norm_params:
                try:
                    forecast = self.denormalize(forecast, norm_params)
                except Exception:
                    pass
        
        return forecast


class ARMAXModel(BaseTimeSeriesModel):
    """
    ARMA модель с экзогенными переменными (ARMAX).
    """
    
    def __init__(self, name: str = "ARMAX", p: int = 1, q: int = 1):
        """
        Args:
            name: Название модели
            p: Порядок авторегрессии
            q: Порядок скользящего среднего
        """
        super().__init__(name)
        self.p = p
        self.q = q
        self.exog_cols = None
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            exog_cols: Optional[list] = None, use_transform: bool = True) -> 'ARMAXModel':
        """
        Обучение ARMAX модели.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            exog_cols: Список названий экзогенных переменных
            use_transform: Использовать ли трансформацию
            
        Returns:
            self
        """
        self.train_data = train_data.copy()
        data = train_data[target_col].copy()
        self.exog_cols = exog_cols
        
        # Подготовка экзогенных переменных
        exog_data = None
        if exog_cols is not None and len(exog_cols) > 0:
            exog_data = train_data[exog_cols].values
        
        if use_transform:
            norm_data, norm_params = self.normalize(data)
            self.transformation_params['normalize'] = norm_params
            transformed_data, lmbda = self.boxcox_transform(norm_data)
            self.transformation_params['boxcox_lambda'] = lmbda
        else:
            transformed_data = data
            
        # ARMAX модель = ARIMA с экзогенными переменными
        model = ARIMA(transformed_data, exog=exog_data, order=(self.p, 0, self.q))
        
        # Use initial_params as starting point if provided (federated learning)
        fit_kwargs = {}
        if self.initial_params:
            try:
                # Convert initial params to numpy array in correct order
                param_names = list(self.initial_params.keys())
                start_params = np.array([self.initial_params.get(name, 0.0) for name in param_names])
                fit_kwargs['start_params'] = start_params
            except Exception:
                pass
        
        self.fitted_model = model.fit(**fit_kwargs)
        self.params = dict(self.fitted_model.params)
        
        return self
    
    def predict(self, steps: int, exog_future: Optional[np.ndarray] = None,
                use_transform: bool = True) -> np.ndarray:
        """
        Прогноз на N шагов вперед.
        
        Args:
            steps: Количество шагов прогноза
            exog_future: Будущие значения экзогенных переменных (steps x n_features)
            use_transform: Использовались ли трансформации
            
        Returns:
            Массив прогнозных значений
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        
        forecast = self.fitted_model.forecast(steps=steps, exog=exog_future)
        
        if use_transform:
            lmbda = self.transformation_params.get('boxcox_lambda') if isinstance(self.transformation_params, dict) else None
            if lmbda is not None:
                try:
                    forecast = self.inverse_boxcox(forecast, lmbda)
                except Exception:
                    pass
            norm_params = self.transformation_params.get('normalize') if isinstance(self.transformation_params, dict) else None
            if norm_params:
                try:
                    forecast = self.denormalize(forecast, norm_params)
                except Exception:
                    pass
        
        return forecast
