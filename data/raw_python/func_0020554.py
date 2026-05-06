def __stop(self):
        """
        Stop the hoverfly process.
        """
        if logging:
            logging.debug("stopping")
        self._process.terminate()
        # communicate means we wait until the process
        # was actually terminated, this removes some
        # warnings in python3
        self._process.communicate()
        self._process = None
        self.FNULL.close()
        self.FNULL = None
        self.__disableProxy()
        # del self._session
        # self._session = None
        self.__rmpid()