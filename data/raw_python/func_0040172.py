def _get_order_clause(archive_table):
    """Returns an ascending order clause on the versioned unique constraint as well as the
    version column.
    """
    order_clause = [
        sa.asc(getattr(archive_table, col_name)) for col_name in archive_table._version_col_names
    ]
    order_clause.append(sa.asc(archive_table.version_id))
    return order_clause