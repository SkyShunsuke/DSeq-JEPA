import logging
import logging.config
from src.utils.log.loggers import WandbLogger, TensorboardLogger, CSVLogger
from src.utils.log.meters import AverageMeter

APP_LOGGER_NAME = "JEPA"
MASTER_LEVEL = "INFO"
OTHERS_LEVEL = "WARNING"

def get_logger(name: str = None) -> logging.Logger:
    parent = logging.getLogger(APP_LOGGER_NAME)
    return parent.getChild(name) if name else parent

def setup_logging(rank, world_size, app_logger=APP_LOGGER_NAME, master_level=MASTER_LEVEL, others_level=OTHERS_LEVEL):
    level_for_this_rank = master_level if rank == 0 else others_level

    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "std": {
                "format": "%(asctime)s [%(levelname)s][GPU|%(rank)s/%(world_size)s] %(filename)s:%(lineno)d (%(funcName)s) - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "std",
                "level": "NOTSET",  
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"logs/app_rank{rank}.log",
                "maxBytes": 10_000_000,
                "backupCount": 3,
                "formatter": "std",
                "level": "NOTSET",
            }
        },
        "loggers": {
            app_logger: {
                "level": level_for_this_rank,
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": { 
            "level": level_for_this_rank,
            "handlers": ["console", "file"]
        }
    }

    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        record.world_size = world_size
        return record
    logging.setLogRecordFactory(record_factory)
    logging.config.dictConfig(LOGGING)