def resetTimeout(self):
        """Reset the timeout count down"""
        if self.__timeoutCall is not None and self.timeOut is not None:
            self.__timeoutCall.reset(self.timeOut)