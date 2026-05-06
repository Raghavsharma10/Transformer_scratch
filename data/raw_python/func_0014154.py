def code(self, code=None, ret_r=False):
        '''
        Args:
            code: (Optional) set code
            ret_r: (Optional) force to return Result. Default value is False
        returns:
            response code(0-success, others-failure) or self
        '''
        if code or ret_r:
            self._code = code
            return self
        return self._code