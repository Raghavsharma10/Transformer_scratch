def api(self, name):
        '''return special API by package's name'''

        assert name, 'name is none'
        if flow.__name__ == name:
            api = flow.FlowApi()
        elif sign.__name__ == name:
            api = sign.SignApi()
        elif sms.__name__ == name:
            api = sms.SmsApi()
        elif tpl.__name__ == name:
            api = tpl.TplApi()
        elif user.__name__ == name:
            api = user.UserApi()
        elif voice.__name__ == name:
            api = voice.VoiceApi()

        assert api, "not found api-" + name

        api._init(self._clnt)
        return api