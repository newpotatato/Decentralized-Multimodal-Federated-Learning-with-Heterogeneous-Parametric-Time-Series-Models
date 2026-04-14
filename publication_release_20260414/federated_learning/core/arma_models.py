"""
Реализация AR, MA, ARMA и ARMAX моделей для прогнозирования временных рядов.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Tuple, Dict, Any, List
import warnings

from base_model import BaseTimeSeriesModel, vector_from_param_names

warnings.filterwarnings('ignore')


def _sanitize_exog_matrix(
    exog: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Drop non-finite / near-constant exogenous columns to avoid singular design matrices."""
    if exog is None:
        return None, None
    arr = np.asarray(exog, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0 or arr.shape[1] == 0:
        return None, None

    # Replace non-finite values before variance checks.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(arr, axis=0)
    keep_idx = np.where(np.isfinite(std) & (std > 1e-10))[0]
    if keep_idx.size == 0:
        return None, None
    return arr[:, keep_idx], keep_idx


def _fit_arima_robust(
    endog,
    order: Tuple[int, int, int],
    *,
    exog: Optional[np.ndarray] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Robust ARIMA fit with retries for common linear algebra failures.

    Retry strategy:
    1) Default constraints, with provided kwargs.
    2) Default constraints, without start_params.
    3) Relaxed stationarity/invertibility, with provided kwargs.
    4) Relaxed stationarity/invertibility, without start_params.
    """
    kwargs0 = dict(fit_kwargs or {})
    attempts = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
    last_exc: Optional[Exception] = None

    for relaxed, drop_start in attempts:
        model_kwargs: Dict[str, Any] = {}
        if relaxed:
            model_kwargs["enforce_stationarity"] = False
            model_kwargs["enforce_invertibility"] = False

        model = ARIMA(endog, exog=exog, order=order, **model_kwargs)
        kwargs = dict(kwargs0)
        if drop_start:
            kwargs.pop("start_params", None)

        try:
            return model.fit(**kwargs)
        except TypeError:
            # Some statsmodels versions reject maxiter in this path.
            kwargs.pop("maxiter", None)
            try:
                return model.fit(**kwargs)
            except Exception as exc:
                last_exc = exc
        except (np.linalg.LinAlgError, ValueError) as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("ARIMA fit failed for unknown reason")


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
        self.fitted_model = _fit_arima_robust(
            transformed_data,
            order=(0, 0, self.q),
            fit_kwargs={},
        )
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
        self.fitted_model = _fit_arima_robust(
            transformed_data,
            order=(self.p, 0, self.q),
            fit_kwargs={},
        )
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
        self._exog_keep_idx: Optional[np.ndarray] = None
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'amt',
            exog_cols: Optional[list] = None, use_transform: bool = True,
            max_iterations: Optional[int] = None) -> 'ARMAXModel':
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
            exog_data_raw = train_data[exog_cols].values
            exog_data, keep_idx = _sanitize_exog_matrix(exog_data_raw)
            self._exog_keep_idx = keep_idx
        else:
            self._exog_keep_idx = None
        
        if use_transform:
            norm_data, norm_params = self.normalize(data)
            self.transformation_params['normalize'] = norm_params
            transformed_data, lmbda = self.boxcox_transform(norm_data)
            self.transformation_params['boxcox_lambda'] = lmbda
        else:
            transformed_data = data
            
        fit_kwargs: Dict[str, Any] = {}
        if self.initial_params:
            try:
                model_for_names = ARIMA(transformed_data, exog=exog_data, order=(self.p, 0, self.q))
                start_params = self._build_start_params(self.initial_params)
                if start_params is None:
                    start_params = vector_from_param_names(
                        list(model_for_names.param_names), self.initial_params
                    )
                if start_params is not None:
                    fit_kwargs['start_params'] = start_params
            except Exception:
                pass
        if max_iterations is not None:
            fit_kwargs['maxiter'] = max(int(max_iterations), 0)

        self.fitted_model = _fit_arima_robust(
            transformed_data,
            order=(self.p, 0, self.q),
            exog=exog_data,
            fit_kwargs=fit_kwargs,
        )
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

        if exog_future is not None and self._exog_keep_idx is not None:
            arr = np.asarray(exog_future, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] >= int(np.max(self._exog_keep_idx)) + 1:
                exog_future = arr[:, self._exog_keep_idx]
            else:
                exog_future = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

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
