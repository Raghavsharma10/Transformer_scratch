def _simple_dispatch(self, name, params):
        """
        Dispatch method
        """
        try:
            # Internal method
            func = self.funcs[name]
        except KeyError:
            # Other method
            pass
        else:
            # Internal method found
            if isinstance(params, (list, tuple)):
                return func(*params)
            else:
                return func(**params)

        # Call the other method outside the except block, to avoid messy logs
        # in case of error
        return self._dispatch_method(name, params)