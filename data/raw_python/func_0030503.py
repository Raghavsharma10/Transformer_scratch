def drop_everything(engine):
    '''Droping all tables and custom types (enums) using `engine`.
    Taken from http://www.sqlalchemy.org/trac/wiki/UsageRecipes/DropEverything

    This method is more robust than `metadata.drop_all(engine)`. B.c. when
    you change a table or a type name, `drop_all` does not consider the old one.
    Thus, DB holds some unused entities.'''
    conn = engine.connect()
    # the transaction only applies if the DB supports
    # transactional DDL, i.e. Postgresql, MS SQL Server
    trans = conn.begin()
    inspector = reflection.Inspector.from_engine(engine)
    metadata = MetaData()
    tbs = []
    all_fks = []
    types = []
    for table_name in inspector.get_table_names():
        fks = []
        for fk in inspector.get_foreign_keys(table_name):
            if not fk['name']:
                continue
            fks.append(ForeignKeyConstraint((), (), name=fk['name']))
        for col in inspector.get_columns(table_name):
            if isinstance(col['type'], SchemaType):
                types.append(col['type'])
        t = Table(table_name, metadata, *fks)
        tbs.append(t)
        all_fks.extend(fks)
    try:
        for fkc in all_fks:
            conn.execute(DropConstraint(fkc))
        for table in tbs:
            conn.execute(DropTable(table))
        for custom_type in types:
            custom_type.drop(conn)
        trans.commit()
    except: # pragma: no cover
        trans.rollback()
        raise