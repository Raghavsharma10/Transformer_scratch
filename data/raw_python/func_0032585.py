def prepare_method_call(self, method, args):
        """
        Wraps a method so that method() will call ``method(*args)`` or ``method(**args)``,
        depending of args type

        :param method: a callable object (method)
        :param args: dict or list with the parameters for the function
        :return: a 'patched' callable
        """
        if self._method_requires_handler_ref(method):
            if isinstance(args, list):
                args = [self] + args
            elif isinstance(args, dict):
                args["handler"] = self

        if isinstance(args, list):
            to_call = partial(method, *args)
        elif isinstance(args, dict):
            to_call = partial(method, **args)
        else:
            raise TypeError(
                "args must be list or dict but got {} instead".format(type(args).__name__))
        return to_call