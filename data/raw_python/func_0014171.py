def pull_reply(self, param=None, must=[APIKEY]):
        '''获取回复短信
        
        参数名 类型 是否必须 描述 示例
        apikey String 是 用户唯一标识 9b11127a9701975c734b8aee81ee3526
        page_size Integer 否 每页个数，最大100个，默认20个 20
        
        Args:
            param:
        Results:
            Result
        '''
        param = {} if param is None else param
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        h = CommonResultHandler(lambda rsp: {VERSION_V1:rsp[SMS_REPLY] if SMS_REPLY in rsp else None, VERSION_V2:rsp}[self.version()])
        return self.path('pull_reply.json').post(param, h, r)