def _ptn2fn(self, pattern):
        ''' Pattern to filename '''
        return [pattern.format(wd=self.working_dir, n=self.__name, mode=self.__mode),
                pattern.format(wd=self.working_dir, n='{}.{}'.format(self.__name, self.__mode), mode=self.__mode)]