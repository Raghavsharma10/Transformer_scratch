def exception(self, e):
        """Log an error messsage.

        :param e:  Exception to log.

        """
        self.logged_exception(e)
        self.logger.exception(e)