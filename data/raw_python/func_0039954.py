def observeState(self, call=None):
        """
        Registers an observer to the any changes.
            The called function should have 2 parameters:
            - previousState,
            - actualState

        :param func call: The function to call.
                          When not given, decorator usage is assumed.
        :return: the function to call once state change.
        :rtype: func
        :raises TypeError: if the called function is not callable

        =================
        How to use it
        =================
        -----------------
        1. Calling the function
        -----------------
            .. code-block:: python
                instance.observeState(functionName)
                instance.observeState(functionName)

                ...
                def functionName(previousState, actualState):

        -----------------
        2. Using Decoration
        -----------------
            .. code-block:: python
                @instance.observeState()
                def functionName(previousState, actualState):

                @instance.observeState()
                def functionName(previousState, actualState):
        """
        def _observe(call):
            self.__observers.add("*", call)
            return call

        if call is not None:
            return _observe(call)
        else:
            return _observe