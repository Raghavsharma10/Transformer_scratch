def find_indexable_materializable(sql, library):
    """
    Parse a statement, then call functions to install, materialize or create indexes for partitions
    referenced in the statement.

    :param sql:
    :param materialize_f:
    :param install_f:
    :param index_f:
    :return:
    """

    derefed, tables, partitions = substitute_vids(library, sql)

    if derefed.lower().startswith('create index') or derefed.lower().startswith('index'):
        parsed = parse_index(derefed)
        return FIMRecord(statement=derefed, indexes=[(parsed.source, tuple(parsed.columns))])

    elif derefed.lower().startswith('materialize'):
        _, vid = derefed.split()
        return FIMRecord(statement=derefed, materialize=set([vid]))

    elif derefed.lower().startswith('install'):
        _, vid = derefed.split()
        return FIMRecord(statement=derefed, install=set([vid]))

    elif derefed.lower().startswith('select'):
        rec = FIMRecord(statement=derefed)
        parsed = parse_select(derefed)

    elif derefed.lower().startswith('drop'):
        return FIMRecord(statement=derefed, drop=derefed)

    elif derefed.lower().startswith('create table'):
        parsed = parse_view(derefed)
        rec = FIMRecord(statement=derefed, drop='DROP TABLE IF EXISTS {};'.format(parsed.name), views=1)

    elif derefed.lower().startswith('create view'):
        parsed = parse_view(derefed)
        rec = FIMRecord(statement=derefed, drop='DROP VIEW IF EXISTS {};'.format(parsed.name), views=1)
    else:
        return FIMRecord(statement=derefed, tables=set(tables), install=set(partitions))

    def partition_aliases(parsed):
        d = {}

        for source in parsed.sources:
            if source.alias:
                d[source.alias] = source.name

        for j in parsed.joins:
            if j.source.alias:
                d[j.source.alias] = j.source.name

        return d

    def indexable_columns(aliases, parsed):

        indexes = []

        for j in parsed.joins:
            if j and j.join_cols:
                for col in j.join_cols:
                    if '.' in col:
                        try:
                            alias, col = col.split('.')
                            if alias:
                                indexes.append((aliases[alias], (col,)))
                        except KeyError:
                            pass

        return indexes

    aliases = partition_aliases(parsed)

    indexes = indexable_columns(aliases, parsed)

    rec.joins = len(parsed.joins)

    install = set(partitions)

    rec.update(tables=tables, install=install, indexes=indexes)

    return rec