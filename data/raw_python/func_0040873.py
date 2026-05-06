def has_constraint(model, engine, *col_names):  # pragma: no cover
    """
    :param model: model class to check
    :param engine: SQLAlchemy engine
    :param col_names: the name of columns which the unique constraint should contain

    :rtype: bool
    :return: True if the given columns are part of a unique constraint on model
    """
    table_name = model.__tablename__
    if engine.dialect.has_table(engine, table_name):
        # Use SQLAlchemy reflection to determine unique constraints
        insp = Inspector.from_engine(engine)
        constraints = itertools.chain(
            (sorted(x['column_names']) for x in insp.get_unique_constraints(table_name)),
            sorted(insp.get_pk_constraint(table_name)['constrained_columns']),
        )
        return sorted(col_names) in constraints
    else:
        # Needed to validate test models pre-creation
        constrained_cols = set()
        for arg in getattr(model, '__table_args__', []):
            if isinstance(arg, UniqueConstraint):
                constrained_cols.update([c.name for c in arg.columns])
        for c in model.__table__.columns:
            if c.primary_key or c.unique:
                constrained_cols.add(c.name)
        return constrained_cols.issuperset(col_names)