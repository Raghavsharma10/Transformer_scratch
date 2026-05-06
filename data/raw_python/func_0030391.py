def login(self, email, password):
        """
        :password: user password md5 digest
        """
        payload = {
            'account': email,
            'password': password
        }
        code, msg, rv = self.request(
            'mtop.alimusic.xuser.facade.xiamiuserservice.login',
            payload
        )
        if code == 'SUCCESS':
            # TODO: 保存 refreshToken 和过期时间等更多信息
            # 根据目前观察，token 过期时间有三年
            accessToken = rv['data']['data']['accessToken']
            self.set_access_token(accessToken)
        return rv