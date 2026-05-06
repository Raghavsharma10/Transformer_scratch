def s(self, *args, **kwargs) -> Partial[Stepwise]:
        """
        Create an unbound prototype of this class, partially applying arguments

        .. code:: python

            @stepwise
            def control(pool: Pool, interval):
                return 10

            pipeline = control.s(interval=20) >> pool

        :note: The partial rules are sealed, and :py:meth:`~.UnboundStepwise.add`
               cannot be called on it.
        """
        return Partial(Stepwise, self.base, *self.rules, *args, **kwargs)