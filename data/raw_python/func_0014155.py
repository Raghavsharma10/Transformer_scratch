def msg(self, msg=None, ret_r=False):
        '''code's message'''
        if msg or ret_r:
            self._msg = msg
            return self
        return self._msg