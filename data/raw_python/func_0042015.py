def __call(self):
        """
        Calls the callback method
        """
        try:
            if self.__callback is not None:
                self.__callback(self.__successes, self.__errors)
        except Exception as ex:
            self.__logger.exception("Error calling back count down "
                                    "handler: %s", ex)
        else:
            self.__called = True