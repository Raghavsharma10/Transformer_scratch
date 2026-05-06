def _instance(self, cls, *args, **kwargs):
        """Return the instance.

        :param cls: the class to create the instance from
        :param args: given to the ``__init__`` method
        :param kwargs: given to the ``__init__`` method
        """
        logger.debug(f'args: {args}, kwargs: {kwargs}')
        return cls(*args, **kwargs)