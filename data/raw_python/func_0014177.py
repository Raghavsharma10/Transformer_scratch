def get_package(self, param=None, must=[APIKEY]):
        '''查询流量包
        
        参数名 类型 是否必须 描述 示例
        apikey String 是 用户唯一标识 9b11127a9701975c734b8aee81ee3526
        carrier String 否 运营商ID 传入该参数则获取指定运营商的流量包， 否则获取所有运营商的流量包 移动：10086 联通：10010 电信：10000
        
        Args:
            param:
        Results:
            Result
        '''
        param = {} if param is None else param
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        h = CommonResultHandler(lambda rsp: {VERSION_V1:rsp[FLOW_PACKAGE] if FLOW_PACKAGE in rsp else None, VERSION_V2:rsp}[self.version()])
        return self.path('get_package.json').post(param, h, r)