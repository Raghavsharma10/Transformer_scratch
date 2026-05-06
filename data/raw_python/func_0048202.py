def _find_config(self, config_file):
        """This method will check if the configuration file "exist"

        If it does NOT then the method will bail calling an IOError.

        :param config_file: ``str``
        :return: ``str``
        """
        msg = ('Configuration file [ %s ] was not found.' % self.filename)
        self.log.fatal(msg)
        raise IOError(msg)