def pop_message(self, till=None):
        """
        RETURN TUPLE (message, payload) CALLER IS RESPONSIBLE FOR CALLING message.delete() WHEN DONE
        DUMMY IMPLEMENTATION FOR DEBUGGING
        """

        if till is not None and not isinstance(till, Signal):
            Log.error("Expecting a signal")
        return Null, self.pop(till=till)