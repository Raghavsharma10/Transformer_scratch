def set_logger(self, logger):
        """
        Set a logger to send debug messages to

        Parameters
        ----------
        logger : `Logger <http://docs.python.org/2/library/logging.html>`_
            A python logger used to get debugging output from this module.
        """
        self.__logger = logger
        self.session.set_logger(self.__logger)