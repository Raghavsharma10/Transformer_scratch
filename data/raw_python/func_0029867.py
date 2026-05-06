def setTimeout(self, period):
        """Change the timeout period

        @type period: C{int} or C{NoneType}
        @param period: The period, in seconds, to change the timeout to, or
        C{None} to disable the timeout.
        """
        prev = self.timeOut
        self.timeOut = period

        if self.__timeoutCall is not None:
            if period is None:
                self.__timeoutCall.cancel()
                self.__timeoutCall = None
            else:
                self.__timeoutCall.reset(period)
        elif period is not None:
            self.__timeoutCall = self.callLater(period, self.__timedOut)

        return prev