def use(self, cookies):
        """
        如果遭遇验证码，用这个接口
        
        :type cookies: str|dict
        :param cookies: cookie字符串或者字典
        :return: self
        """
        self.cookies = dict([item.split('=', 1) for item in re.split(r'; *', cookies)]) \
            if isinstance(cookies, str) else cookies
        self.flush()
        self.persist()
        return self