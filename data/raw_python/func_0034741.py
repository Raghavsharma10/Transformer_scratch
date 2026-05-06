def hook(self, hook_name):
        """A decorator that's used to register a new hook handler.

        :param hook_name: the event to handle
        """
        def wrapper(fn):
            self.register_hook(hook_name, fn)
            return fn
        return wrapper