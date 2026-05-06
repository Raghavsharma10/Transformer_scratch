def raise_if(self, exception, message, *args, **kwargs):
        """
        If current exception has smaller priority than minimum, subclass of
        this class only warns user, otherwise normal exception will be raised.
        """
        if issubclass(exception, self.minimum_defect):
            raise exception(*args, **kwargs)
        warn(message, SyntaxWarning, *args, **kwargs)