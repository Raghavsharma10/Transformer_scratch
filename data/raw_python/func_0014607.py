def get_request(cls):
        """
        Get the HTTPRequest object from thread storage or from a callee by searching
        each frame in the call stack.
        """
        request = cls.get_global('request')
        if request:
            return request
        try:
            stack = inspect.stack()
        except IndexError:
            # in some cases this may return an index error
            # (pyc files dont match py files for example)
            return
        for frame, _, _, _, _, _ in stack:
            if 'request' in frame.f_locals:
                if isinstance(frame.f_locals['request'], HttpRequest):
                    request = frame.f_locals['request']
                    cls.set_global('request', request)
                    return request