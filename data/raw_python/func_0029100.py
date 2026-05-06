def flush(self):
        """
        更新会话信息，主要是ck, user_alias
        """
        if 'dbcl2' not in self.cookies:
            return
        r = self.req(API_ACCOUNT_HOME)
        if RE_SESSION_EXPIRE.search(r.url):
            return self.expire()
        self.cookies.update(dict(r.cookies))
        self.user_alias = slash_right(r.url)
        self.logger.debug('flush with user_alias <%s>' % self.user_alias)
        return