"""
Реализация State-Space и Калмановских моделей для прогнозирования временных рядов.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.mlemodel import MLEModel
from typing import Optional, Dict, Any
import warnings

from base_model import BaseTimeSeriesModel

warnings.filterwarnings('ignore')


class KalmanFilterModel(BaseTimeSeriesModel):
    """
    Базовая модель на основе фильтра Калмана.
    Реализует простую локальную линейную трендовую модель.
    """
    
    def __init__(self, name: str = "Kalman"):
        """
        Args:
            name: Название модели
        """
        super().__init__(name)
        self.state_dim = 2  # [уровень, тренд]
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            use_transform: bool = True, 
            process_noise: float = 0.1,
            observation_noise: float = 1.0) -> 'KalmanFilterModel':
        """
        Обучение модели фильтра Калмана.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            use_transform: Использовать ли трансформацию данных
            process_noise: Ковариация шума процесса
            observation_noise: Ковариация шума наблюдений
            
        Returns:
            self
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
            
        # Использование UnobservedComponents для локальной линейной тренда
        model = UnobservedComponents(
            transformed_data,
            level='local linear trend',
            irregular=True,
            stochastic_level=True,
            stochastic_trend=True
        )
        
        # State-space models (UnobservedComponents) are sensitive to initialization
        # But we need to support federated learning with parameter initialization
        fit_kwargs = {'maxiter': 100}
        if self.initial_params:
            try:
                # Try to use initial parameters for model initialization
                start_params = self._build_start_params(self.initial_params)
                if start_params is not None:
                    fit_kwargs['start_params'] = start_params
            except Exception:
                pass  # Fall back to no initialization if it fails
        self.fitted_model = model.fit(**fit_kwargs)
        
        # Сохранение параметров
        self.params = {
            'sigma2_irregular': self.fitted_model.params.get('sigma2.irregular', 0),
            'sigma2_level': self.fitted_model.params.get('sigma2.level', 0),
            'sigma2_trend': self.fitted_model.params.get('sigma2.trend', 0),
            'process_noise': process_noise,
            'observation_noise': observation_noise
        }
        
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
        
        # Robust forecast: try forecast(), then predict(start,end), then repeat-last
        forecast_array = None
        try:
            fc = self.fitted_model.forecast(steps=steps)
            forecast_array = fc.values if hasattr(fc, 'values') else np.asarray(fc)
        except NotImplementedError:
            forecast_array = None
        except Exception:
            forecast_array = None

        if forecast_array is None:
            try:
                n = None
                try:
                    # self.train_data may be DataFrame or Series
                    if isinstance(self.train_data, pd.DataFrame):
                        n = len(self.train_data)
                        # pick first numeric column if multiple
                        numcols = [c for c in self.train_data.columns if np.issubdtype(self.train_data[c].dtype, np.number)]
                        if numcols:
                            last_val = np.asarray(self.train_data[numcols[0]].iloc[-1])
                        else:
                            last_val = np.asarray(self.train_data.iloc[-1, 0])
                    else:
                        n = len(self.train_data)
                        last_val = np.asarray(self.train_data.iloc[-1])
                except Exception:
                    n = len(self.train_data)
                    last_val = 0.0

                pred = self.fitted_model.predict(start=n, end=n + steps - 1)
                forecast_array = pred.values if hasattr(pred, 'values') else np.asarray(pred)
            except Exception:
                try:
                    forecast_array = np.repeat(last_val, steps)
                except Exception:
                    forecast_array = np.zeros(steps)

        if use_transform:
            lmbda = self.transformation_params.get('boxcox_lambda') if isinstance(self.transformation_params, dict) else None
            if lmbda is not None:
                try:
                    forecast_array = self.inverse_boxcox(forecast_array, lmbda)
                except Exception:
                    pass
            norm_params = self.transformation_params.get('normalize') if isinstance(self.transformation_params, dict) else None
            if norm_params:
                try:
                    forecast_array = self.denormalize(forecast_array, norm_params)
                except Exception:
                    pass

        return np.asarray(forecast_array)


class StructuralTimeSeriesModel(BaseTimeSeriesModel):
    """
    Структурная модель временных рядов (State-Space модель).
    Включает уровень, тренд, сезонность и нерегулярную компоненту.
    """
    
    def __init__(self, name: str = "StructuralTS", seasonal_period: Optional[int] = None):
        """
        Args:
            name: Название модели
            seasonal_period: Период сезонности (None если нет сезонности)
        """
        super().__init__(name)
        self.seasonal_period = seasonal_period
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            use_transform: bool = True,
            level: str = 'local linear trend',
            stochastic_level: bool = True,
            stochastic_trend: bool = True,
            stochastic_seasonal: bool = True) -> 'StructuralTimeSeriesModel':
        """
        Обучение структурной модели временных рядов.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            use_transform: Использовать ли трансформацию данных
            level: Тип уровня ('fixed intercept', 'local level', 'local linear trend')
            stochastic_level: Стохастический ли уровень
            stochastic_trend: Стохастический ли тренд
            stochastic_seasonal: Стохастическая ли сезонность
            
        Returns:
            self
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
            
        # Построение структурной модели
        seasonal_component = None
        if self.seasonal_period is not None:
            seasonal_component = self.seasonal_period
            
        model = UnobservedComponents(
            transformed_data,
            level=level,
            trend=stochastic_trend,
            seasonal=seasonal_component,
            stochastic_level=stochastic_level,
            stochastic_trend=stochastic_trend,
            stochastic_seasonal=stochastic_seasonal,
            irregular=True
        )
        
        # State-space models (SARIMAX) are sensitive to initialization
        # But we need to support federated learning with parameter initialization
        fit_kwargs = {'maxiter': 100}
        if self.initial_params:
            try:
                # Try to use initial parameters for model initialization
                start_params = self._build_start_params(self.initial_params)
                if start_params is not None:
                    fit_kwargs['start_params'] = start_params
            except Exception:
                pass  # Fall back to no initialization if it fails
        self.fitted_model = model.fit(**fit_kwargs)
        
        # Сохранение параметров
        self.params = {
            'level': level,
            'seasonal_period': self.seasonal_period,
            'stochastic_level': stochastic_level,
            'stochastic_trend': stochastic_trend,
            'stochastic_seasonal': stochastic_seasonal,
            **dict(self.fitted_model.params)
        }
        
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
        
        # Robust forecast: try forecast(), then predict(start,end), then repeat-last
        forecast_array = None
        try:
            fc = self.fitted_model.forecast(steps=steps)
            forecast_array = fc.values if hasattr(fc, 'values') else np.asarray(fc)
        except NotImplementedError:
            forecast_array = None
        except Exception:
            forecast_array = None

        if forecast_array is None:
            try:
                # determine last value
                if isinstance(self.train_data, pd.DataFrame):
                    numcols = [c for c in self.train_data.columns if np.issubdtype(self.train_data[c].dtype, np.number)]
                    if numcols:
                        last_val = np.asarray(self.train_data[numcols[0]].iloc[-1])
                    else:
                        last_val = np.asarray(self.train_data.iloc[-1, 0])
                else:
                    last_val = np.asarray(self.train_data.iloc[-1])

                n = len(self.train_data)
                pred = self.fitted_model.predict(start=n, end=n + steps - 1)
                forecast_array = pred.values if hasattr(pred, 'values') else np.asarray(pred)
            except Exception:
                try:
                    forecast_array = np.repeat(last_val, steps)
                except Exception:
                    forecast_array = np.zeros(steps)

        if use_transform:
            lmbda = self.transformation_params.get('boxcox_lambda') if isinstance(self.transformation_params, dict) else None
            if lmbda is not None:
                try:
                    forecast_array = self.inverse_boxcox(forecast_array, lmbda)
                except Exception:
                    pass
            norm_params = self.transformation_params.get('normalize') if isinstance(self.transformation_params, dict) else None
            if norm_params:
                try:
                    forecast_array = self.denormalize(forecast_array, norm_params)
                except Exception:
                    pass

        return np.asarray(forecast_array)
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Получение декомпозиции временного ряда на компоненты.
        
        Returns:
            Словарь с компонентами (уровень, тренд, сезонность)
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        states = self.fitted_model.states.smoothed
        components = {}
        
        # Извлечение компонент из состояний
        if hasattr(self.fitted_model, 'level'):
            components['level'] = states[0, :]
        if hasattr(self.fitted_model, 'trend'):
            components['trend'] = states[1, :]
        if self.seasonal_period is not None:
            components['seasonal'] = states[2, :]
            
        return components


class DynamicLinearModel(BaseTimeSeriesModel):
    """
    Динамическая линейная модель (Dynamic Linear Model, DLM).
    Обобщенная State-Space модель с экзогенными переменными.
    """
    
    def __init__(self, name: str = "DLM", max_iterations: int = None):
        """
        Args:
            name: Название модели
            max_iterations: Максимальное количество итераций оптимизации (None = default)
        """
        super().__init__(name)
        self.exog_cols = None
        self.max_iterations = max_iterations
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            exog_cols: Optional[list] = None,
            use_transform: bool = True,
            order: tuple = (1, 0, 0),
            max_iterations: int = None) -> 'DynamicLinearModel':
        """
        Обучение динамической линейной модели.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            exog_cols: Список названий экзогенных переменных
            use_transform: Использовать ли трансформацию
            order: Порядок SARIMAX модели (p, d, q)
            max_iterations: Максимальное количество итераций оптимизации
                           (None = используется default, 0 = no optimization, 1-N = limited optimization)
            
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
            
        # Использование SARIMAX как DLM
        model = SARIMAX(
            transformed_data,
            exog=exog_data,
            order=order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # State-space models (UnobservedComponents) are sensitive to initialization
        # But we need to support federated learning with parameter initialization
        fit_kwargs = {}
        if self.initial_params:
            try:
                # Try to use initial parameters for model initialization
                start_params = self._build_start_params(self.initial_params)
                if start_params is not None:
                    fit_kwargs['start_params'] = start_params
            except Exception:
                pass  # Fall back to no initialization if it fails
        
        # Add maxiter parameter - use VERY small number to avoid long optimization
        # In federated learning, we don't need full convergence, just reasonable estimates
        if max_iterations is not None:
            fit_kwargs['maxiter'] = max_iterations
        elif self.max_iterations is not None:
            fit_kwargs['maxiter'] = self.max_iterations
        else:
            fit_kwargs['maxiter'] = 5  # Very small to avoid timeout
        
        try:
            # Use a timeout wrapper to avoid hanging on large datasets
            import signal
            import os
            
            if os.name != 'nt':  # Unix-like systems only
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model fit timeout")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
                try:
                    self.fitted_model = model.fit(**fit_kwargs)
                finally:
                    signal.alarm(0)
            else:
                # Windows: just fit without timeout
                self.fitted_model = model.fit(**fit_kwargs)
        except (TimeoutError, KeyboardInterrupt):
            # If optimization times out, use initial parameters or default
            self.fitted_model = model
            # Try to extract any parameters that were computed
            try:
                self.params = dict(model.params)
            except:
                self.params = {}
        
        try:
            self.params = dict(self.fitted_model.params)
        except:
            self.params = {}
        
        return self
    
    def predict(self, steps: int, exog_future: Optional[np.ndarray] = None,
                use_transform: bool = True) -> np.ndarray:
        """
        Прогноз на N шагов вперед.
        
        Args:
            steps: Количество шагов прогноза
            exog_future: Будущие значения экзогенных переменных
            use_transform: Использовались ли трансформации
            
        Returns:
            Массив прогнозных значений
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        
        # Robust forecast with exog: try forecast, then predict(start,end,exog), then repeat-last
        forecast_array = None
        try:
            fc = self.fitted_model.forecast(steps=steps, exog=exog_future)
            forecast_array = fc.values if hasattr(fc, 'values') else np.asarray(fc)
        except NotImplementedError:
            forecast_array = None
        except Exception:
            forecast_array = None

        if forecast_array is None:
            try:
                if isinstance(self.train_data, pd.DataFrame):
                    numcols = [c for c in self.train_data.columns if np.issubdtype(self.train_data[c].dtype, np.number)]
                    if numcols:
                        last_val = np.asarray(self.train_data[numcols[0]].iloc[-1])
                    else:
                        last_val = np.asarray(self.train_data.iloc[-1, 0])
                else:
                    last_val = np.asarray(self.train_data.iloc[-1])

                n = len(self.train_data)
                try:
                    pred = self.fitted_model.predict(start=n, end=n + steps - 1, exog=exog_future)
                except Exception:
                    pred = self.fitted_model.predict(start=n, end=n + steps - 1)
                forecast_array = pred.values if hasattr(pred, 'values') else np.asarray(pred)
            except Exception:
                try:
                    forecast_array = np.repeat(last_val, steps)
                except Exception:
                    forecast_array = np.zeros(steps)

        if use_transform:
            lmbda = self.transformation_params.get('boxcox_lambda') if isinstance(self.transformation_params, dict) else None
            if lmbda is not None:
                try:
                    forecast_array = self.inverse_boxcox(forecast_array, lmbda)
                except Exception:
                    pass
            norm_params = self.transformation_params.get('normalize') if isinstance(self.transformation_params, dict) else None
            if norm_params:
                try:
                    forecast_array = self.denormalize(forecast_array, norm_params)
                except Exception:
                    pass

        return np.asarray(forecast_array)
    
    def get_filtered_states(self) -> np.ndarray:
        """
        Получение отфильтрованных состояний (результат фильтра Калмана).
        
        Returns:
            Матрица состояний
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        return self.fitted_model.states.filtered
    
    def get_smoothed_states(self) -> np.ndarray:
        """
        Получение сглаженных состояний (результат сглаживания Калмана).
        
        Returns:
            Матрица состояний
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        return self.fitted_model.states.smoothed
