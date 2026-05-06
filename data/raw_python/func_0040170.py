def _get_conditions_list(table, conds, archive=True):
    """This function returns a list of list of == conditions on sqlalchemy columns given conds.
    This should be treated as an or of ands.

    :param table: the user table model class which inherits from
        savage.models.SavageModelMixin
    :param conds: a list of dictionaries of key value pairs where keys are column names and
        values are conditions to be placed on the column.
    :param archive: If true, the condition is with columns from the archive table. Else its from
        the user table.
    """
    if conds is None:
        conds = []

    all_conditions = []
    for cond in conds:
        if len(cond) != len(table.version_columns):
            raise ValueError('Conditions must specify all unique constraints.')

        conditions = []
        t = table.ArchiveTable if archive else table

        for col_name, value in cond.iteritems():
            if col_name not in table.version_columns:
                raise ValueError('{} is not one of the unique columns <{}>'.format(
                    col_name, ','.join(table.version_columns)
                ))
            conditions.append(getattr(t, col_name) == value)
        all_conditions.append(conditions)
    return all_conditions