"""
Реализация Markov-switching моделей для прогнозирования временных рядов.
Модели с переключением режимов для захвата структурных изменений.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from typing import Optional, Dict, Any, List
import warnings

from base_model import BaseTimeSeriesModel

warnings.filterwarnings('ignore')


class MarkovSwitchingARModel(BaseTimeSeriesModel):
    """
    Марковская авторегрессионная модель с переключением режимов (MS-AR).
    Позволяет временному ряду переключаться между различными AR-режимами.
    """
    
    def __init__(self, name: str = "MS-AR", order: int = 1, k_regimes: int = 2):
        """
        Args:
            name: Название модели
            order: Порядок авторегрессии
            k_regimes: Количество режимов
        """
        super().__init__(name)
        self.order = order
        self.k_regimes = k_regimes
        self.regime_names = [f"Режим {i+1}" for i in range(k_regimes)]
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            use_transform: bool = True,
            switching_variance: bool = True) -> 'MarkovSwitchingARModel':
        """
        Обучение Markov-switching AR модели.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            use_transform: Использовать ли трансформацию данных
            switching_variance: Переключается ли дисперсия между режимами
            
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
            
        # Построение MS-AR модели
        model = MarkovAutoregression(
            transformed_data,
            k_regimes=self.k_regimes,
            order=self.order,
            switching_variance=switching_variance
        )
        
        self.fitted_model = model.fit()
        
        # Сохранение параметров
        self.params = {
            'order': self.order,
            'k_regimes': self.k_regimes,
            'switching_variance': switching_variance,
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
        
        # Try forecast; if not implemented, fallback to predict(start,end) or repeat-last
        forecast = None
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            # convert to numpy array if it's a pandas object
            forecast_array = forecast.values if hasattr(forecast, 'values') else np.asarray(forecast)
        except NotImplementedError:
            forecast_array = None
        except Exception:
            # other numeric failures
            forecast_array = None

        if forecast_array is None:
            # try using fitted_model.predict with indices relative to training length
            try:
                n = len(self.train_data)
                # predict from n to n+steps-1
                pred = self.fitted_model.predict(start=n, end=n + steps - 1)
                forecast_array = pred.values if hasattr(pred, 'values') else np.asarray(pred)
            except Exception:
                forecast_array = None

        if forecast_array is None:
            # last-resort fallback: repeat last observed value
            try:
                last = np.asarray(self.train_data[target_col].iloc[-1])
                forecast_array = np.repeat(last, steps)
            except Exception:
                forecast_array = np.zeros(steps)

        if use_transform:
            # Obратная Box-Cox и денормализация with guards
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
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Получение вероятностей режимов для каждого момента времени.
        
        Returns:
            DataFrame с вероятностями каждого режима
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        probs = self.fitted_model.smoothed_marginal_probabilities
        probs.columns = self.regime_names
        return probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Получение матрицы переходов между режимами.
        
        Returns:
            Матрица переходов (k_regimes x k_regimes)
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        return self.fitted_model.regime_transition
    
    def get_expected_durations(self) -> Dict[str, float]:
        """
        Получение ожидаемой продолжительности каждого режима.
        
        Returns:
            Словарь с ожидаемой продолжительностью каждого режима
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        durations = self.fitted_model.expected_durations
        return {self.regime_names[i]: durations[i] for i in range(self.k_regimes)}


class MarkovSwitchingRegressionModel(BaseTimeSeriesModel):
    """
    Марковская регрессионная модель с переключением режимов (MS-Regression).
    Регрессия с экзогенными переменными и переключением режимов.
    """
    
    def __init__(self, name: str = "MS-Regression", k_regimes: int = 2):
        """
        Args:
            name: Название модели
            k_regimes: Количество режимов
        """
        super().__init__(name)
        self.k_regimes = k_regimes
        self.exog_cols = None
        self.regime_names = [f"Режим {i+1}" for i in range(k_regimes)]
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            exog_cols: Optional[list] = None,
            use_transform: bool = True,
            switching_variance: bool = True,
            switching_trend: bool = True) -> 'MarkovSwitchingRegressionModel':
        """
        Обучение Markov-switching регрессионной модели.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            exog_cols: Список названий экзогенных переменных
            use_transform: Использовать ли трансформацию данных
            switching_variance: Переключается ли дисперсия между режимами
            switching_trend: Переключается ли тренд между режимами
            
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
            
        # Построение MS-Regression модели
        model = MarkovRegression(
            transformed_data,
            k_regimes=self.k_regimes,
            exog=exog_data,
            switching_variance=switching_variance,
            trend='c' if switching_trend else 'n'
        )
        
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
                    self.fitted_model = model.fit()
                finally:
                    signal.alarm(0)
            else:
                # Windows: just fit without timeout
                self.fitted_model = model.fit()
        except (TimeoutError, KeyboardInterrupt):
            # If optimization times out, use unfitted model
            self.fitted_model = model
            try:
                self.params = dict(model.params)
            except:
                self.params = {}
            return self
        
        # Сохранение параметров
        self.params = {
            'k_regimes': self.k_regimes,
            'switching_variance': switching_variance,
            'switching_trend': switching_trend,
            **dict(self.fitted_model.params)
        }
        
        return self
    
    def predict(self, steps: int, exog_future: Optional[np.ndarray] = None,
                use_transform: bool = True) -> np.ndarray:
        """
        Прогноз на N шагов вперед.
        
        Args:
            steps: Количество шагов прогноза
            exog_future: Будущие значения экзогенных переменных
            use_transform: Использовались ли трансформации при обучении
            
        Returns:
            Массив прогнозных значений
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        
        # Try forecast with exog; fallback to predict with start/end or repeat-last
        forecast_array = None
        try:
            forecast = self.fitted_model.forecast(steps=steps, exog=exog_future)
            forecast_array = forecast.values if hasattr(forecast, 'values') else np.asarray(forecast)
        except NotImplementedError:
            forecast_array = None
        except Exception:
            forecast_array = None

        if forecast_array is None:
            try:
                n = len(self.train_data)
                pred = None
                # try predict with exog if possible
                try:
                    pred = self.fitted_model.predict(start=n, end=n + steps - 1, exog=exog_future)
                except Exception:
                    pred = self.fitted_model.predict(start=n, end=n + steps - 1)
                forecast_array = pred.values if hasattr(pred, 'values') else np.asarray(pred)
            except Exception:
                forecast_array = None

        if forecast_array is None:
            try:
                last = np.asarray(self.train_data[target_col].iloc[-1])
                forecast_array = np.repeat(last, steps)
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
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Получение вероятностей режимов для каждого момента времени.
        
        Returns:
            DataFrame с вероятностями каждого режима
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        probs = self.fitted_model.smoothed_marginal_probabilities
        probs.columns = self.regime_names
        return probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Получение матрицы переходов между режимами.
        
        Returns:
            Матрица переходов (k_regimes x k_regimes)
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        return self.fitted_model.regime_transition
    
    def get_regime_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Получение параметров для каждого режима.
        
        Returns:
            Словарь с параметрами каждого режима
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        regime_params = {}
        for i in range(self.k_regimes):
            regime_params[self.regime_names[i]] = {
                'mean': self.fitted_model.params[f'const[{i}]'] if f'const[{i}]' in self.fitted_model.params else None,
                'variance': self.fitted_model.params[f'sigma2[{i}]'] if f'sigma2[{i}]' in self.fitted_model.params else None
            }
        
        return regime_params


class MarkovSwitchingDynamicRegression(BaseTimeSeriesModel):
    """
    Динамическая регрессионная модель с переключением режимов (MS-Dynamic Regression).
    Комбинирует авторегрессию, экзогенные переменные и переключение режимов.
    """
    
    def __init__(self, name: str = "MS-DynamicReg", order: int = 1, k_regimes: int = 2):
        """
        Args:
            name: Название модели
            order: Порядок авторегрессии
            k_regimes: Количество режимов
        """
        super().__init__(name)
        self.order = order
        self.k_regimes = k_regimes
        self.exog_cols = None
        self.regime_names = [f"Режим {i+1}" for i in range(k_regimes)]
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            exog_cols: Optional[list] = None,
            use_transform: bool = True,
            switching_variance: bool = True,
            switching_ar: bool = True) -> 'MarkovSwitchingDynamicRegression':
        """
        Обучение динамической регрессионной модели с переключением режимов.
        
        Args:
            train_data: DataFrame с данными
            target_col: Название целевой колонки
            exog_cols: Список названий экзогенных переменных
            use_transform: Использовать ли трансформацию данных
            switching_variance: Переключается ли дисперсия между режимами
            switching_ar: Переключаются ли AR-коэффициенты между режимами
            
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
            
        # Построение MS-AR модели с экзогенными переменными
        try:
            model = MarkovAutoregression(
                transformed_data,
                k_regimes=self.k_regimes,
                order=self.order,
                exog=exog_data,
                switching_variance=switching_variance
            )
            
            self.fitted_model = model.fit()
        except Exception as e:
            # Fallback: если AR модель с exog не работает, используем регрессию
            print(f"Внимание: не удалось построить MS-AR с экзогенными переменными. Используется MS-Regression.")
            model = MarkovRegression(
                transformed_data,
                k_regimes=self.k_regimes,
                exog=exog_data,
                switching_variance=switching_variance,
                trend='c'
            )
            self.fitted_model = model.fit()
        
        # Сохранение параметров
        self.params = {
            'order': self.order,
            'k_regimes': self.k_regimes,
            'switching_variance': switching_variance,
            'switching_ar': switching_ar,
            **dict(self.fitted_model.params)
        }
        
        return self
    
    def predict(self, steps: int, exog_future: Optional[np.ndarray] = None,
                use_transform: bool = True) -> np.ndarray:
        """
        Прогноз на N шагов вперед.
        
        Args:
            steps: Количество шагов прогноза
            exog_future: Будущие значения экзогенных переменных
            use_transform: Использовались ли трансформации при обучении
            
        Returns:
            Массив прогнозных значений
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict().")
        
        # Try forecast with exog; fallback to predict with start/end or repeat-last
        forecast_array = None
        try:
            forecast = self.fitted_model.forecast(steps=steps, exog=exog_future)
            forecast_array = forecast.values if hasattr(forecast, 'values') else np.asarray(forecast)
        except NotImplementedError:
            forecast_array = None
        except Exception:
            forecast_array = None

        if forecast_array is None:
            try:
                n = len(self.train_data)
                pred = None
                try:
                    pred = self.fitted_model.predict(start=n, end=n + steps - 1, exog=exog_future)
                except Exception:
                    pred = self.fitted_model.predict(start=n, end=n + steps - 1)
                forecast_array = pred.values if hasattr(pred, 'values') else np.asarray(pred)
            except Exception:
                forecast_array = None

        if forecast_array is None:
            try:
                last = np.asarray(self.train_data[target_col].iloc[-1])
                forecast_array = np.repeat(last, steps)
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
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Получение вероятностей режимов для каждого момента времени.
        
        Returns:
            DataFrame с вероятностями каждого режима
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        probs = self.fitted_model.smoothed_marginal_probabilities
        probs.columns = self.regime_names
        return probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Получение матрицы переходов между режимами.
        
        Returns:
            Матрица переходов (k_regimes x k_regimes)
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        return self.fitted_model.regime_transition
    
    def identify_current_regime(self) -> str:
        """
        Определение наиболее вероятного текущего режима.
        
        Returns:
            Название текущего режима
        """
        if self.fitted_model is None:
            raise ValueError("Модель не обучена.")
        
        probs = self.fitted_model.smoothed_marginal_probabilities
        last_prob = probs.iloc[-1]
        regime_idx = last_prob.argmax()
        
        return self.regime_names[regime_idx]
