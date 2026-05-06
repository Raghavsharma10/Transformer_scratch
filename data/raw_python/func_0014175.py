def tpl_send(self, param, must=[APIKEY, MOBILE, TPL_ID, TPL_VALUE]):
        '''指定模板发送 only v1 deprecated
        
        参数名 类型 是否必须 描述 示例
        apikey String 是 用户唯一标识 9b11127a9701975c734b8aee81ee3526
        mobile String 是 接收的手机号 15205201314
        tpl_id Long 是 模板id 1
        tpl_value String 是 变量名和变量值对。请先对您的变量名和变量值分别进行urlencode再传递。使用参考：代码示例。
        注：变量名和变量值都不能为空 模板： 【#company#】您的验证码是#code#。 最终发送结果： 【云片网】您的验证码是1234。
        tplvalue=urlencode("#code#") + "=" + urlencode("1234") + "&amp;" +
        urlencode("#company#") + "=" + urlencode("云片网"); 若您直接发送报文请求则使用下面这种形式
        tplvalue=urlencode(urlencode("#code#") + "=" + urlencode("1234") + "&amp;" +
        urlencode("#company#") + "=" + urlencode("云片网"));
        extend String 否 扩展号。默认不开放，如有需要请联系客服申请 001
        uid String 否 用户自定义唯一id。最大长度不超过256的字符串。 默认不开放，如有需要请联系客服申请 10001
        
        Args:
            param:
        Results:
            Result
        '''
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        h = CommonResultHandler(lambda rsp: {VERSION_V1:rsp.get(RESULT)}[self.version()])
        return self.path('tpl_send.json').post(param, h, r)