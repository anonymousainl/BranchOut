import logging
import os
from datetime import datetime
from pathlib import Path
import functools

def setup_logging(log_level: str = "INFO"):
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    root_logger.addHandler(console_handler)

    debug_file = logs_dir / "app_debug.log"
    debug_handler = logging.FileHandler(debug_file, mode='a', encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_handler)

    api_file = logs_dir / "api_calls.log"
    api_handler = logging.FileHandler(api_file, mode='a', encoding='utf-8')
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(detailed_formatter)

    api_logger_instance = logging.getLogger("api_calls")
    api_logger_instance.setLevel(logging.INFO)
    api_logger_instance.addHandler(api_handler)
    api_logger_instance.propagate = False

    error_file = logs_dir / "errors.log"
    error_handler = logging.FileHandler(error_file, mode='a', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    agent_file = logs_dir / "agent_activity.log"
    agent_handler = logging.FileHandler(agent_file, mode='a', encoding='utf-8')
    agent_handler.setLevel(logging.INFO)
    agent_handler.setFormatter(detailed_formatter)

    agent_logger_instance = logging.getLogger("agent_activity")
    agent_logger_instance.setLevel(logging.INFO)
    agent_logger_instance.addHandler(agent_handler)
    agent_logger_instance.propagate = False

    logging.info(f"Logging initialized. Logs directory: {logs_dir.absolute()}")

app_logger = logging.getLogger(__name__)
api_logger = logging.getLogger("api_calls")
agent_logger = logging.getLogger("agent_activity")

class LoggerContext:
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.level = level
        self.original_level = None

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(getattr(logging, self.level.upper()))
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_function_call(func):

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        app_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            app_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            app_logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        app_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            app_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            app_logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper