def _method_is_async_generator(self, method):
        """
        Given a simple callable or a callable wrapped in funtools.partial, determines
        if it was wrapped with the :py:func:`gemstone.async_method` decorator.
        
        :param method:
        :return:
        """
        if hasattr(method, "func"):
            func = method.func
        else:
            func = method

        return getattr(func, "_is_coroutine", False)