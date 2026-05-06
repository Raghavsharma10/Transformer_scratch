def get_log_config(component, handlers, level='DEBUG', path='/var/log/vfine/'):
    """Return a log config for django project."""
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s][%(threadName)s]' +
                          '[%(name)s.%(funcName)s():%(lineno)d] %(message)s'
            },
            'color': {
                '()': 'shaw.log.SplitColoredFormatter',
                'format': "%(asctime)s " +
                          "%(log_color)s%(bold)s[%(levelname)s]%(reset)s" +
                          "[%(threadName)s][%(name)s.%(funcName)s():%(lineno)d] " +
                          "%(blue)s%(message)s"
            }
        },
        'handlers': {
            'debug': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': path + component + '.debug.log',
                'maxBytes': 1024 * 1024 * 1024,
                'backupCount': 5,
                'formatter': 'standard',
            },
            'color': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': path + component + '.color.log',
                'maxBytes': 1024 * 1024 * 1024,
                'backupCount': 5,
                'formatter': 'color',
            },
            'info': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': path + component + '.info.log',
                'maxBytes': 1024 * 1024 * 1024,
                'backupCount': 5,
                'formatter': 'standard',
            },
            'error': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': path + component + '.error.log',
                'maxBytes': 1024 * 1024 * 100,
                'backupCount': 5,
                'formatter': 'standard',
            },
            'console': {
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
        },
        'loggers': {
            'django': {
                'handlers': handlers,
                'level': 'INFO',
                'propagate': False
            },
            'django.request': {
                'handlers': handlers,
                'level': 'INFO',
                'propagate': False,
            },
            '': {
                'handlers': handlers,
                'level': level,
                'propagate': False
            },
        }
    }
    return config