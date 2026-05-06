def update_voice_notify(self, param, must=[APIKEY, TPL_ID, TPL_CONTENT]):
        '''修改语音通知模版
        
        注意：模板成功修改之后需要重新审核才能使用！同时提醒您如果修改了变量，务必重新测试，以免替换出错!
        参数：
        参数名    类型    是否必须    描述    示例
        apikey    String    是    用户唯一标识    9b11127a9701975c734b8aee81ee3526
        tpl_id    Long    是    模板id，64位长整形。指定id时返回id对应的模板。未指定时返回所有模板    9527
        tpl_content    String    是    模板id，64位长整形。指定id时返回id对应的模板。未指定时返回所有模板模板内容    您的验证码是#code#
        
        Args:
            param:  
        Results:
            Result
        '''
        r = self.verify_param(param, must)
        if not r.is_succ():
            return r
        h = CommonResultHandler(lambda rsp: {VERSION_V2:rsp}[self.version()])
        return self.path('update_voice_notify.json').post(param, h, r)