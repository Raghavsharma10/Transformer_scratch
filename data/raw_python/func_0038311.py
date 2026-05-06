def add_column(self, table, name='ID', data_type='int(11)', after_col=None, null=False, primary_key=False):
        """Add a column to an existing table."""
        location = 'AFTER {0}'.format(after_col) if after_col else 'FIRST'
        null_ = 'NULL' if null else 'NOT NULL'
        comment = "COMMENT 'Column auto created by mysql-toolkit'"
        pk = 'AUTO_INCREMENT PRIMARY KEY {0}'.format(comment) if primary_key else ''
        query = 'ALTER TABLE {0} ADD COLUMN {1} {2} {3} {4} {5}'.format(wrap(table), name, data_type, null_, pk,
                                                                        location)
        self.execute(query)
        self._printer("\tAdded column '{0}' to '{1}' {2}".format(name, table, '(Primary Key)' if primary_key else ''))
        return name