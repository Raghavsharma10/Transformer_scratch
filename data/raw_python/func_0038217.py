def create_table(self, name, data, columns=None, add_pk=True):
        """Generate and execute a create table query by parsing a 2D dataset"""
        # TODO: Issue occurs when bool values exist in data
        # Remove if the table exists
        if name in self.tables:
            self.drop(name)

        # Set headers list
        if not columns:
            columns = data[0]

        # Validate data shape
        for row in data:
            assert len(row) == len(columns)

        # Create dictionary of column types
        col_types = {columns[i]: sql_column_type([d[i] for d in data], prefer_int=True, prefer_varchar=True)
                     for i in range(0, len(columns))}

        # Join column types into SQL string
        cols = ''.join(['\t{0} {1},\n'.format(name, type_) for name, type_ in col_types.items()])[:-2] + '\n'
        statement = 'CREATE TABLE {0} ({1}{2})'.format(name, '\n', cols)
        self.execute(statement)
        if add_pk:
            self.set_primary_key_auto()
        return True