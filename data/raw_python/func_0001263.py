def logger(self):
        ''' Lazy logger '''
        if self.__logger is None:
            self.__logger = logging.getLogger(self.__name)
        return self.__logger