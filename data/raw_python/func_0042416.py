def withArgs(self, *args, **kwargs): #pylint: disable=invalid-name
        """
        Adds a condition for when the stub is called. When the condition is met, a special
        return value can be returned. Adds the specified argument(s) into the condition list.

        For example, when the stub function is called with argument 1, it will return "#":
            stub.withArgs(1).returns("#")

        Without returns/throws at the end of the chain of functions, nothing will happen.
        For example, in this case, although 1 is in the condition list, nothing will happen:
            stub.withArgs(1)

        Return:
            a SinonStub object (able to be chained)
        """
        cond_args = args if len(args) > 0 else None
        cond_kwargs = kwargs if len(kwargs) > 0 else None
        return _SinonStubCondition(copy=self._copy, cond_args=cond_args, cond_kwargs=cond_kwargs, oncall=self._oncall)