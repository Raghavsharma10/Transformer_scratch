def build_row_dict(cls, row, dialect, deleted=False, user_id=None, use_dirty=True):
        """
        Builds a dictionary of archive data from row which is suitable for insert.

        NOTE: If `deleted` is False, version ID will be set to an AsIs SQL construct.

        :param row: instance of :class:`~SavageModelMixin`
        :param dialect: :py:class:`~sqlalchemy.engine.interfaces.Dialect`
        :param deleted: whether or not the row is deleted (defaults to False)
        :param user_id: ID of user that is performing the update on this row (defaults to None)
        :param use_dirty: whether to use the dirty fields from row or not (defaults to True)
        :return: a dictionary of archive table column names to values, suitable for insert
        :rtype: dict
        """
        data = {
            'data': row.to_archivable_dict(dialect, use_dirty=use_dirty),
            'deleted': deleted,
            'updated_at': datetime.now(),
            'version_id': current_version_sql(as_is=True) if deleted else row.version_id
        }
        for col_name in row.version_columns:
            data[col_name] = utils.get_column_attribute(row, col_name, use_dirty=use_dirty)
        if user_id is not None:
            data['user_id'] = user_id
        return data