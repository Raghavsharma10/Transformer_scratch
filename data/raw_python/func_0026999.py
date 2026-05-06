def log_backend_action(action=None):
    """ Logging for backend method.

    Expects django model instance as first argument.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(self, instance, *args, **kwargs):
            action_name = func.func_name.replace('_', ' ') if action is None else action

            logger.debug('About to %s `%s` (PK: %s).', action_name, instance, instance.pk)
            result = func(self, instance, *args, **kwargs)
            logger.debug('Action `%s` was executed successfully for `%s` (PK: %s).',
                         action_name, instance, instance.pk)
            return result
        return wrapped
    return decorator