def detail(self, detail=None, ret_r=False):
        '''code's detail'''
        if detail or ret_r:
            self._detail = detail
            return self
        return self._detail