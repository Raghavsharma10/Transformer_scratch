def get(self, param=None, must=[APIKEY]):
        '''查账户信息
        
        参数名 类型 是否必须 描述 示例
        apikey String 是 用户唯一标识 9b11127a9701975c734b8aee81ee3526
        
        Args:
            param: (Optional) 
        Results:
            Result
        '''
        param = {} if param is None else param
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        handle = CommonResultHandler(lambda rsp: {VERSION_V1:rsp.get(USER), VERSION_V2:rsp}[self.version()])
        return self.path('get.json').post(param, handle, r)