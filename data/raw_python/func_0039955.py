def observeElements(self, what, call=None):
        """
        Registers an observer function to a specific state field or
            list of state fields.
            The function to call should have 2 parameters:
            - previousValue,
            -actualValue

        :param what: name of the state field or names of the
                     state field to observe.
        :type what: str | array
        :param func call: The function to call. When not given,
                          decorator usage is assumed.
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
            instance.observeFields("FieldName", functionName)
            instance.observeFields(["FieldName1","FieldName2"], functionName)

            ...
            def functionName(previousState, actualState):

        -----------------
        2. Using Decoration
        -----------------
        .. code-block:: python
            @instance.observeFields("FieldName")
            def functionName(previousValue, actualValue):

            @instance.observeFields(["FieldName1","FieldName2"])
            def functionName(previousValue, actualValue):
        """
        def _observe(call):
            self.__observers.add(what, call)
            return call

        toEvaluate = []
        if isinstance(what, str):
            toEvaluate.append(what)
        else:
            toEvaluate = what

        if not self.areObservableElements(toEvaluate):
            msg = 'Could not find observable element named "{0}" in {1}'
            raise ValueError(msg.format(what, self.__class__))

        if call is not None:
            return _observe(call)
        else:
            return _observe