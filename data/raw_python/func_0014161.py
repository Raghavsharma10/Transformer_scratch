def post(self, param, h, r):
        '''
        Args:
            param: request parameters
            h: ResultHandler
            r: YunpianApiResult
        '''
        try:
            rsp = self.client().post(self.uri(), param)
            # print(rsp)
            return self.result(rsp, h, r)
        except ValueError as err:
            return h.catch_exception(err, r)