def require_ajax_logged_in(func):
    """Check if ajax API is logged in and login if not
    """
    @functools.wraps(func)
    def inner_func(self, *pargs, **kwargs):
        if not self._ajax_api.logged_in:
            logger.info('Logging into AJAX API for required meta method')
            if not self.has_credentials:
                raise ApiLoginFailure(
                    'Login is required but no credentials were provided')
            self._ajax_api.User_Login(name=self._state['username'],
                password=self._state['password'])
        return func(self, *pargs, **kwargs)
    return inner_func