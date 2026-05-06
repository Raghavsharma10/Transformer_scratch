def persist(self):
        """
        持久化会话信息
        """
        with open(self.persist_file, 'w+') as f:
            json.dump({
                'cookies': self.cookies,
                'user_alias': self.user_alias,
            }, f, indent=2)
            self.logger.debug('persist session to <%s>' % self.persist_file)