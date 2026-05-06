def _preprocess_sqlite_view(asql_query, library, backend, connection):
    """ Finds view or materialized view in the asql query and converts it to create table/insert rows.

    Note:
        Assume virtual tables for all partitions already created.

    Args:
        asql_query (str): asql query
        library (ambry.Library):
        backend (SQLiteBackend):
        connection (apsw.Connection):

    Returns:
        str: valid sql query containing create table and insert into queries if asql_query contains
            'create materialized view'. If asql_query does not contain 'create materialized view' returns
            asql_query as is.
    """

    new_query = None

    if 'create materialized view' in asql_query.lower() or 'create view' in asql_query.lower():

        logger.debug(
            '_preprocess_sqlite_view: materialized view found.\n    asql query: {}'
            .format(asql_query))

        view = parse_view(asql_query)

        tablename = view.name.replace('-', '_').lower().replace('.', '_')
        create_query_columns = {}
        for column in view.columns:
            create_query_columns[column.name] = column.alias

        ref_to_partition_map = {}  # key is ref found in the query, value is Partition instance.
        alias_to_partition_map = {}  # key is alias of ref found in the query, value is Partition instance.

        # collect sources from select statement of the view.
        for source in view.sources:
            partition = library.partition(source.name)
            ref_to_partition_map[source.name] = partition
            if source.alias:
                alias_to_partition_map[source.alias] = partition

        # collect sources from joins of the view.
        for join in view.joins:
            partition = library.partition(join.source.name)
            ref_to_partition_map[join.source.name] = partition
            if join.source.alias:
                alias_to_partition_map[join.source.alias] = partition

        # collect and convert columns.
        TYPE_MAP = {
            'int': 'INTEGER',
            'float': 'REAL',
            six.binary_type.__name__: 'TEXT',
            six.text_type.__name__: 'TEXT',
            'date': 'DATE',
            'datetime': 'TIMESTAMP WITHOUT TIME ZONE'
        }
        column_types = []
        column_names = []
        for column in view.columns:
            if '.' in column.name:
                source_alias, column_name = column.name.split('.')
            else:
                # TODO: Test that case.
                source_alias = None
                column_name = column.name

            # find column specification in the mpr file.
            if source_alias:
                partition = alias_to_partition_map[source_alias]
                for part_column in partition.datafile.reader.columns:
                    if part_column['name'] == column_name:
                        sqlite_type = TYPE_MAP.get(part_column['type'])
                        if not sqlite_type:
                            raise Exception(
                                'Do not know how to convert {} to sql column.'
                                .format(column['type']))

                        column_types.append(
                            '    {} {}'
                            .format(column.alias if column.alias else column.name, sqlite_type))
                        column_names.append(column.alias if column.alias else column.name)

        column_types_str = ',\n'.join(column_types)
        column_names_str = ', '.join(column_names)

        create_query = 'CREATE TABLE IF NOT EXISTS {}(\n{});'.format(tablename, column_types_str)

        # drop 'create materialized view' part
        _, select_part = asql_query.split(view.name)
        select_part = select_part.strip()
        assert select_part.lower().startswith('as')

        # drop 'as' keyword
        select_part = select_part.strip()[2:].strip()
        assert select_part.lower().strip().startswith('select')

        # Create query to copy data from mpr to just created table.
        copy_query = 'INSERT INTO {table}(\n{columns})\n  {select}'.format(
            table=tablename, columns=column_names_str, select=select_part)
        if not copy_query.strip().lower().endswith(';'):
            copy_query = copy_query + ';'
        new_query = '{}\n\n{}'.format(create_query, copy_query)
    logger.debug(
        '_preprocess_sqlite_view: preprocess finished.\n    asql query: {}\n\n    new query: {}'
        .format(asql_query, new_query))


    return new_query or asql_query