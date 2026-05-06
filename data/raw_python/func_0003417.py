def result(self):
        '''
        :return: None if the result is not ready, the result from set_result, or raise the exception
                 from set_exception. If the result can be None, it is not possible to tell if the result is
                 available; use done() to determine that.
        '''
        try:
            r = getattr(self, '_result')
        except AttributeError:
            return None
        else:
            if hasattr(self, '_exception'):
                raise self._exception
            else:
                return r