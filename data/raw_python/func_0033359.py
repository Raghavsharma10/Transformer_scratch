def get_changed_oids(self, last_update=None):
        '''
        Returns a list of object ids of those objects that have changed since
        `mtime`. This method expects that the changed objects can be
        determined based on the `delta_mtime` property of the cube which
        specifies the field name that carries the time of the last change.

        This method is expected to be overriden in the cube if it is not
        possible to use a single field to determine the time of the change and
        if another approach of determining the oids is available. In such
        cubes the `delta_mtime` property is expected to be set to `True`.

        If `delta_mtime` evaluates to False then this method is not expected
        to be used.

        :param mtime: datetime string used as 'change since date'
        '''
        mtime_columns = self.lconfig.get('delta_mtime', [])
        if not (mtime_columns and last_update):
            return []
        mtime_columns = str2list(mtime_columns)
        where = []
        for _column in mtime_columns:
            _sql = "%s >= %s" % (_column, last_update)
            where.append(_sql)
        return self.sql_get_oids(where)