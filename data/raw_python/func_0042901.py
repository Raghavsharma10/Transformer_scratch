def _open(self):
        """ DO NOT USE THIS UNLESS YOU close() FIRST"""
        try:
            self.db = connect(
                host=self.settings.host,
                port=self.settings.port,
                user=coalesce(self.settings.username, self.settings.user),
                passwd=coalesce(self.settings.password, self.settings.passwd),
                db=coalesce(self.settings.schema, self.settings.db),
                read_timeout=coalesce(self.settings.read_timeout, (EXECUTE_TIMEOUT / 1000) - 10 if EXECUTE_TIMEOUT else None, 5*60),
                charset=u"utf8",
                use_unicode=True,
                ssl=coalesce(self.settings.ssl, None),
                cursorclass=cursors.SSCursor
            )
        except Exception as e:
            if self.settings.host.find("://") == -1:
                Log.error(
                    u"Failure to connect to {{host}}:{{port}}",
                    host=self.settings.host,
                    port=self.settings.port,
                    cause=e
                )
            else:
                Log.error(u"Failure to connect.  PROTOCOL PREFIX IS PROBABLY BAD", e)
        self.cursor = None
        self.partial_rollback = False
        self.transaction_level = 0
        self.backlog = []  # accumulate the write commands so they are sent at once
        if self.readonly:
            self.begin()