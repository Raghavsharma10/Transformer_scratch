def cache(time=0, byUser=None):
    """Decorator to specify the number of seconds the result should be cached, and if cache can be shared between users
    :param time:   Cache timeout in seconds
    :param byUser: False by default, if true, cache is shared among users
    """
    def real_decorator(function):
        function.pi_api_cache_time = time
        function.pi_api_cache_by_user = byUser
        return function
    return real_decorator