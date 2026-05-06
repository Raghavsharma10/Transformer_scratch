def get_new_oids(self):
        '''
        Returns a list of unique oids that have not been extracted yet.

        Essentially, a diff of distinct oids in the source database
        compared to cube.
        '''
        table = self.lconfig.get('table')
        _oid = self.lconfig.get('_oid')
        if is_array(_oid):
            _oid = _oid[0]  # get the db column, not the field alias
        last_id = self.container.get_last_field(field='_oid')
        ids = []
        if last_id:
            try:  # try to convert to integer... if not, assume unicode value
                last_id = float(last_id)
                where = "%s.%s > %s" % (table, _oid, last_id)
            except (TypeError, ValueError):
                where = "%s.%s > '%s'" % (table, _oid, last_id)
            ids = self.sql_get_oids(where)
        return ids