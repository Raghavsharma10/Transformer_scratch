def load(self):
        """
        加载会话信息
        """
        if not os.path.isfile(self.persist_file):
            return
        with open(self.persist_file, 'r') as f:
            cfg = json.load(f) or {}
            self.cookies = cfg.get('cookies', {})
            self.user_alias = cfg.get('user_alias') or None
            self.logger.debug('load session for <%s> from <%s>' % (self.user_alias, self.persist_file))