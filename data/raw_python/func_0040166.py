def delete(table, session, conds):
    """Performs a hard delete on a row, which means the row is deleted from the Savage
    table as well as the archive table.

    :param table: the model class which inherits from
        :class:`~savage.models.user_table.SavageModelMixin` and specifies the model
        of the user table from which we are querying
    :param session: a sqlalchemy session with connections to the database
    :param conds: a list of dictionary of key value pairs where keys are columns in the table
        and values are values the column should take on. If specified, this query will
        only return rows where the columns meet all the conditions. The columns specified
        in this dictionary must be exactly the unique columns that versioning pivots around.
    """
    with session.begin_nested():
        archive_conds_list = _get_conditions_list(table, conds)
        session.execute(
            sa.delete(table.ArchiveTable, whereclause=_get_conditions(archive_conds_list))
        )
        conds_list = _get_conditions_list(table, conds, archive=False)
        session.execute(
            sa.delete(table, whereclause=_get_conditions(conds_list))
        )