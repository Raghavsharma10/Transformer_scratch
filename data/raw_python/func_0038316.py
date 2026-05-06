def modify_column(self, table, name, new_name=None, data_type=None, null=None, default=None):
        """Modify an existing column."""
        existing_def = self.get_schema_dict(table)[name]

        # Set column name
        new_name = new_name if new_name is not None else name

        # Set data type
        if not data_type:
            data_type = existing_def['Type']

        # Set NULL
        if null is None:
            null_ = 'NULL' if existing_def['Null'].lower() == 'yes' else 'NOT NULL'
        else:
            null_ = 'NULL' if null else 'NOT NULL'

        default = 'DEFAULT {0}'.format(default if default else null_)

        query = 'ALTER TABLE {0} CHANGE {1} {2} {3} {4} {5}'.format(wrap(table), wrap(name), wrap(new_name), data_type,
                                                                           null_, default)
        self.execute(query)
        self._printer('\tModified column {0}'.format(name))