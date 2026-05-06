def single_send(self, param, must=[APIKEY, MOBILE, TEXT]):
        '''单条发送
        
        参数名 类型 是否必须 描述 示例
        apikey String 是 用户唯一标识 9b11127a9701975c734b8aee81ee3526
        mobile String 是
        接收的手机号；仅支持单号码发送；国际号码需包含国际地区前缀号码，格式必须是"+"号开头("+"号需要urlencode处理，否则会出现格式错误)，国际号码不以"+"开头将被认为是中国地区的号码
        （针对国际短信，mobile参数会自动格式化到E.164格式，可能会造成传入mobile参数跟后续的状态报告中的号码不一致。E.164格式说明，参见：
        https://en.wikipedia.org/wiki/E.164） 国内号码：15205201314
        国际号码：urlencode("+93701234567");
        text String 是 短信内容 【云片网】您的验证码是1234
        extend String 否 扩展号。默认不开放，如有需要请联系客服申请 001
        uid String 否 该条短信在您业务系统内的ID，比如订单号或者短信发送记录的流水号。填写后发送状态返回值内将包含这个ID
        默认不开放，如有需要请联系客服申请 10001
        callback_url String 否
        本条短信状态报告推送地址。短信发送后将向这个地址推送短信发送报告。"后台-系统设置-数据推送与获取”可以做批量设置。如果后台已经设置地址的情况下，单次请求内也包含此参数，将以请求内的推送地址为准。
        http://your_receive_url_address
        
        Args:
            param: 
        Results:
            Result
        '''
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        h = CommonResultHandler(lambda rsp: {VERSION_V2:rsp}[self.version()])
        return self.path('single_send.json').post(param, h, r)