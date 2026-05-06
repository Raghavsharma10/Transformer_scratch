def login(self, username, password):
        """
        登录
        
        :type username: str
        :param username: 用户名（手机号或者邮箱）
        
        :type password: str
        :param password: 密码
        """
        r0 = self.req(API_HOME)
        time.sleep(1)
        cookies = dict(r0.cookies)
        data = {
            'source': 'index_nav',
            'form_email': username,
            'form_password': password,
            'remember': 'on',
        }
        r1 = self.req(API_ACCOUNT_LOGIN, method='post', data=data)
        cookies.update(dict(r1.cookies))
        [cookies.update(dict(r.cookies)) for r in r1.history]
        if 'dbcl2' not in cookies:
            raise Exception('Authorization failed for <%s>: %s' % (username, r1.url))
        cookies.update(dict(r1.cookies))
        self.logger.info('login with username <%s>' % username)
        self.use(cookies)
        return self