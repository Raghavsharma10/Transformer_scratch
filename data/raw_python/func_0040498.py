def _eval(self, m: EvalParam) -> object:
        """
        Evaluate m returning the method / function invocation or value.  Kind of like a static method
        :param m: object to evaluate
        :return: return
        """
        if inspect.ismethod(m) or inspect.isroutine(m):
            return m()
        elif inspect.isfunction(m):
            return m(self) if len(inspect.signature(m)) > 0 else m()
        else:
            return m