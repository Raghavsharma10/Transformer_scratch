def require_android_logged_in(func):
    """Check if andoid API is logged in and login if not, implies
    `require_session_started`
    """
    @functools.wraps(func)
    @require_session_started
    def inner_func(self, *pargs, **kwargs):
        if not self._android_api.logged_in:
            logger.info('Logging into android API for required meta method')
            if not self.has_credentials:
                raise ApiLoginFailure(
                    'Login is required but no credentials were provided')
            self._android_api.login(account=self._state['username'],
                password=self._state['password'])
        return func(self, *pargs, **kwargs)
    return inner_func