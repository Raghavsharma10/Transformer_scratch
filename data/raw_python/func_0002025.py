def register_parser_callback(self, func):
        """
        Register a callback function that is called after self.iocs and self.ioc_name is populated.

        This is intended for use by subclasses that may have additional parsing requirements.

        :param func:  A callable function.  This should accept a single input, which will be an IOC class.
        :return:
        """
        if hasattr(func, '__call__'):
            self.parser_callback = func
            log.debug('Set callback to {}'.format(func))
        else:
            raise TypeError('Provided function is not callable: {}'.format(func))