def s(cls: Type[C], *args, **kwargs) -> Partial[C]:
        """
        Create an unbound prototype of this class, partially applying arguments

        .. code:: python

            controller = Controller.s(interval=20)

            pipeline = controller(rate=10) >> pool
        """
        return Partial(cls, *args, **kwargs)