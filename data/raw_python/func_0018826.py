def POST_evaluate(self) -> None:
        """Evaluate any valid Python expression with the *HydPy* server
        process and get its result.

        Method |HydPyServer.POST_evaluate| serves to test and debug, primarily.
        The main documentation on module |servertools| explains its usage.
        """
        for name, value in self._inputs.items():
            result = eval(value)
            self._outputs[name] = objecttools.flatten_repr(result)