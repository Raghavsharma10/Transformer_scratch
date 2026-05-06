def register_helper(self, func):
        """
        A helper is a task that is not directly exposed to
        the command line
        :param func: registers func as a helper
        :return: invalid accessor
        """

        self._helper_names.add(func.__name__)
        return self.register(func)