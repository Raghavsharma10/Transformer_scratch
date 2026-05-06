def get_last_field(self, field, table=None):
        '''Shortcut for querying to get the last field value for
        a given owner, cube.

        :param field: field name to query
        '''
        field = field if is_array(field) else [field]
        table = self.get_table(table, except_=False)
        if table is None:
            last = None
        else:
            is_defined(field, 'field must be defined!')
            last = self.find(table=table, fields=field, scalar=True,
                             sort=field, limit=1, descending=True,
                             date='~', default_fields=False)
        logger.debug("last %s.%s: %s" % (table, list2str(field), last))
        return last