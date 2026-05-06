def onCall(self, n): #pylint: disable=invalid-name
        """
        Adds a condition for when the stub is called. When the condition is met, a special
        return value can be returned. Adds the specified call number into the condition
        list.

        For example, when the stub function is called the second time, it will return "#":
            stub.onCall(1).returns("#")

        Without returns/throws at the end of the chain of functions, nothing will happen.
        For example, in this case, although 2 is in the condition list, nothing will happen:
            stub.onCall(2)

        Args:
            n: integer, the call # for which we want a special return value.
               The first call has an index of 0.

        Return:
            a SinonStub object (able to be chained)
        """
        cond_oncall = n + 1
        return _SinonStubCondition(copy=self._copy, oncall=cond_oncall, cond_args=self._cond_args, cond_kwargs=self._cond_kwargs)