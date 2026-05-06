def _get_create_query(partition, tablename, include=None):
        """ Creates and returns `CREATE TABLE ...` sql statement for given mprows.

        Args:
            partition (orm.Partition):
            tablename (str): name of the table in the return create query.
            include (list of str, optional): list of columns to include to query.

        Returns:
            str: create table query.

        """
        TYPE_MAP = {
            'int': 'INTEGER',
            'float': 'REAL',
            six.binary_type.__name__: 'TEXT',
            six.text_type.__name__: 'TEXT',
            'date': 'DATE',
            'datetime': 'TIMESTAMP WITHOUT TIME ZONE'
        }
        columns_types = []
        if not include:
            include = []
        for column in sorted(partition.datafile.reader.columns, key=lambda x: x['pos']):
            if include and column['name'] not in include:
                continue
            sqlite_type = TYPE_MAP.get(column['type'])
            if not sqlite_type:
                raise Exception('Do not know how to convert {} to sql column.'.format(column['type']))
            columns_types.append('    "{}" {}'.format(column['name'], sqlite_type))
        columns_types_str = ',\n'.join(columns_types)
        query = 'CREATE TABLE IF NOT EXISTS {}(\n{})'.format(tablename, columns_types_str)
        return query