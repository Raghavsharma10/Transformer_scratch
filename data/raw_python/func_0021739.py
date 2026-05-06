def require_session_started(func):
    """Check if API sessions are started and start them if not
    """
    @functools.wraps(func)
    def inner_func(self, *pargs, **kwargs):
        if not self.session_started:
            logger.info('Starting session for required meta method')
            self.start_session()
        return func(self, *pargs, **kwargs)
    return inner_func