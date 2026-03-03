import asyncio
import functools
import random
import time
from typing import Callable, Any
from src.logger import app_logger

def infinite_retry_with_backoff(max_wait: int = 120,max_retries: int = 5,base_delay: float = 1.0,exponential_base: float = 2.0):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            last_exception = None
            while True:
                attempt += 1
                try:
                    app_logger.debug(f"[{func.__name__}] Попыт(ка) {attempt}")
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        app_logger.info(f"[{func.__name__}] Урааа он сделал это! Попытка {attempt}")
                    return result
                except Exception as e:
                    last_exception = e
                    if max_retries != -1 and attempt >= max_retries:
                        app_logger.error(f"[{func.__name__}] Ашибка! На {attempt} попытке. " 
                                         f"Last error: {e}")
                        raise
                    delay = min(max_wait,base_delay * (exponential_base ** (attempt - 1)) + random.uniform(0, 1))
                    app_logger.warning(
                        f"[{func.__name__}] Попытка номер {attempt} ошиблась: {e}. "
                        f"Попробуем ещё разок через {delay:.2f} ...")
                    await asyncio.sleep(delay)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            last_exception = None
            while True:
                attempt += 1
                try:
                    app_logger.debug(f"[{func.__name__}] Попыт(ка) {attempt}")
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        app_logger.info(f"[{func.__name__}] Урааа он сделал это! Попытка {attempt}")
                    return result

                except Exception as e:
                    last_exception = e
                    if max_retries != -1 and attempt >= max_retries:
                        app_logger.error(f"[{func.__name__}]  Ашибка! На {attempt} попытке. "
                            f"Last error: {e}")
                        raise
                    delay = min(max_wait,base_delay * (exponential_base ** (attempt - 1)) + random.uniform(0, 1))
                    app_logger.warning(f"[{func.__name__}]  Попытка номер {attempt} ошиблась: {e}. "
                        f"Попробуем ещё разок через {delay:.2f} ...")
                    time.sleep(delay)
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def rate_limit(calls_per_minute: int = 60):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                app_logger.debug(f"[{func.__name__}]Спим (рейт лимит) {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                app_logger.debug(f"[{func.__name__}] Спим (рейт лимит) {sleep_time:.2f}s")
                time.sleep(sleep_time)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def timeout(seconds: int):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                app_logger.error(f"[{func.__name__}] Таймаут  {seconds} сек")
                raise TimeoutError(f"{func.__name__} таймаут после {seconds} сек")
        return wrapper
    return decorator