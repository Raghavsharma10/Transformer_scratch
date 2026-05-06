def request(self, action, payload, timeout=3):
        """
        虾米 API 请求流程：

        1. 获取一个 token：随便访问一个网页，服务端会 set cookie。
           根据观察，这个 token 一般是 7 天过期
        2. 对请求签名：见 _sign_payload 方法
        3. 发送请求
        """
        if self._req_token is None:  # 获取 token
            self._fetch_token()

        url = _gen_url(action)
        params = self._sign_payload(payload)
        response = self.http.get(url, params=params,
                                 cookies=self._cookies.get('cookie'),
                                 timeout=timeout)
        rv = response.json()
        code, msg = rv['ret'][0].split('::')
        # app id 和 key 不匹配，一般应该不会出现这种情况
        if code == 'FAIL_SYS_PARAMINVALID_ERROR':
            raise RuntimeError('Xiami api app id and key not match.')
        elif code == 'FAIL_SYS_TOKEN_EXOIRED':  # 刷新 token
            self._fetch_token()
        elif code == 'FAIL_BIZ_GLOBAL_NEED_LOGIN':
            # TODO: 单独定义一个 Exception
            raise RuntimeError('Xiami api need access token.')
        else:
            if code != 'SUCCESS':
                logger.warning('Xiami request failed:: '
                               'req_action: {}, req_payload: {}\n'
                               'response: {}'
                               .format(action, payload, rv))
            return code, msg, rv