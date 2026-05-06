def sql_get_oids(self, where=None):
        '''
        Query source database for a distinct list of oids.
        '''
        table = self.lconfig.get('table')
        db = self.lconfig.get('db_schema_name') or self.lconfig.get('db')
        _oid = self.lconfig.get('_oid')
        if is_array(_oid):
            _oid = _oid[0]  # get the db column, not the field alias
        sql = 'SELECT DISTINCT %s.%s FROM %s.%s' % (table, _oid, db, table)
        if where:
            where = [where] if isinstance(where, basestring) else list(where)
            sql += ' WHERE %s' % ' OR '.join(where)
        result = sorted([r[_oid] for r in self._load_sql(sql)])
        return result