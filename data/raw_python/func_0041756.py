def remove(self, what, call):
        """
        remove an observer

        what: (string | array) state fields to observe
        call: (function) when not given, decorator usage is assumed.
            The call function should have 2 parameters:
            - previousValue,
            - actualValue

        """
        type = observerTypeEnum.typeOf(what)
        self._observers.remove({
                                    "observing": what,
                                    "type": type,
                                    "call": call
                                 })