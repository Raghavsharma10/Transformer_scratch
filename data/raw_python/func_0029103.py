def logout(self):
        """
        登出会话
        
        :return: self
        """
        self.req(API_ACCOUNT_LOGOUT % self.ck())
        self.cookies = {}
        self.user_alias = None
        self.persist()