def _sqla_postgresql(self, uri, version=None,
                         isolation_level="READ COMMITTED"):
        '''
        expected uri form:
        postgresql+psycopg2://%s:%s@%s:%s/%s' % (
            username, password, host, port, db)
        '''
        isolation_level = isolation_level or "READ COMMITTED"
        kwargs = dict(isolation_level=isolation_level)
        # FIXME: version of postgresql < 9.2 don't have pg.JSON!
        # check and use JSONTypedLite instead
        # override default dict and list column types
        types = {list: pg.ARRAY, tuple: pg.ARRAY, set: pg.ARRAY,
                 dict: JSONDict, datetime: UTCEpoch}
        self.type_map.update(types)
        bs = self.config['batch_size']
        # 999 batch_size is default for sqlite, postgres handles more at once
        self.config['batch_size'] = 5000 if bs == 999 else bs
        self._lock_required = False
        # default schema name is 'public' for postgres
        dsn = self.config['db_schema']
        self.config['db_schema'] = dsn or 'public'
        return uri, kwargs