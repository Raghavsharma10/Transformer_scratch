def count(self, param, must=[APIKEY, START_TIME, END_TIME]):
        '''统计短信条数
        
        参数名 类型 是否必须 描述 示例
        apikey String 是 用户唯一标识 9b11127a9701975c734b8aee81ee3526
        start_time String 是 短信发送开始时间 2013-08-11 00:00:00
        end_time String 是 短信发送结束时间 2013-08-12 00:00:00
        mobile String 否 需要查询的手机号 15205201314
        page_num Integer 否 页码，默认值为1 1
        page_size Integer 否 每页个数，最大100个 20
        
        Args:
            param:
        Results:
            Result
        '''
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        h = CommonResultHandler(lambda rsp: int(rsp[TOTAL]) if TOTAL in rsp else 0)
        return self.path('count.json').post(param, h, r)