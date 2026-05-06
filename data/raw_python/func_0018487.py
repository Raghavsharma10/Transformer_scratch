def set_table(genome, table, table_name, connection_string, metadata):
    """
    alter the table to work between different
    dialects
    """
    table = Table(table_name, genome._metadata, autoload=True,
                    autoload_with=genome.bind, extend_existing=True)

    #print "\t".join([c.name for c in table.columns])
    # need to prefix the indexes with the table name to avoid collisions
    for i, idx in enumerate(table.indexes):
        idx.name = table_name + "." + idx.name + "_ix" + str(i)

    cols = []
    for i, col in enumerate(table.columns):
        # convert mysql-specific types to varchar
        #print col.name, col.type, isinstance(col.type, ENUM)
        if isinstance(col.type, (LONGBLOB, ENUM)):

            if 'sqlite' in connection_string:
                col.type = VARCHAR()
            elif 'postgres' in connection_string:
                if isinstance(col.type, ENUM):
                    #print dir(col)
                    col.type = PG_ENUM(*col.type.enums, name=col.name,
                        create_type=True)
                else:
                    col.type = VARCHAR()
        elif str(col.type) == "VARCHAR" \
                and ("mysql" in connection_string \
                or "postgres" in connection_string):
            if col.type.length is None:
                col.type.length = 48 if col.name != "description" else None
        if not "mysql" in connection_string:
            if str(col.type).lower().startswith("set("):
                col.type = VARCHAR(15)
        cols.append(col)

    table = Table(table_name, genome._metadata, *cols,
            autoload_replace=True, extend_existing=True)

    return table