def recharge(self, param, must=[APIKEY, MOBILE, SN]):
        '''充值流量
        
        参数名 类型 是否必须 描述 示例
        apikey String 是 用户唯一标识 9b11127a9701975c734b8aee81ee3526
        mobile String 是 接收的手机号（仅支持大陆号码） 15205201314
        sn String 是 流量包的唯一ID 点击查看 1008601
        callback_url String 否 本条流量充值的状态报告推送地址 http://your_receive_url_address
        encrypt String 否 加密方式 使用加密 tea (不再使用)
        _sign String 否 签名字段 参考使用加密 393d079e0a00912335adfe46f4a2e10f (不再使用)
        
        Args:
            param:
        Results:
            Result
        '''
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        h = CommonResultHandler(lambda rsp: {VERSION_V1:rsp[RESULT] if RESULT in rsp else None, VERSION_V2:rsp}[self.version()])
        return self.path('recharge.json').post(param, h, r)