def copy(self, klass=None):
        """Create a new instance of the current chain.
        """
        chain = (
            klass if klass else self.__class__
        )(*self._args, **self._kwargs)
        chain._tokens = self._tokens.copy()
        return chain