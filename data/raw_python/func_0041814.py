def to_archivable_dict(self, dialect, use_dirty=True):
        """
        :param dialect: a :py:class:`~sqlalchemy.engine.interfaces.Dialect` corresponding to the \
            SQL dialect being used.
        :param use_dirty: whether to make a dict of the fields as they stand, or the fields \
            before the row was updated

        :return: a dictionary of key value pairs representing this row.
        :rtype: dict
        """
        return {
            cn: utils.get_column_attribute(self, c, use_dirty=use_dirty, dialect=dialect)
            for c, cn in utils.get_column_keys_and_names(self)
            if c not in self.ignore_columns
        }