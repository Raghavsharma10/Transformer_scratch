def substitute_vids(library, statement):
    """ Replace all of the references to tables and partitions with their vids.

    This is a bit of a hack -- it ought to work with the parser, but instead it just looks for
    common SQL tokens that indicate an identifier.

    :param statement: an sqlstatement. String.
    :return: tuple: new_statement, set of table vids, set of partition vids.
    """
    from ambry.identity import ObjectNumber, TableNumber, NotObjectNumberError
    from ambry.orm.exc import NotFoundError

    try:
        stmt_str = statement.to_unicode()
    except AttributeError:
        stmt_str = statement

    parts = stmt_str.strip(';').split()

    new_parts = []

    tables = set()
    partitions = set()

    while parts:
        token = parts.pop(0).strip()
        if token.lower() in ('from', 'join', 'materialize', 'install'):
            ident = parts.pop(0).strip(';')
            new_parts.append(token)

            try:
                obj_number = ObjectNumber.parse(token)
                if isinstance(obj_number, TableNumber):
                    table = library.table(ident)
                    tables.add(table.vid)
                    new_parts.append(table.vid)
                else:
                    # Do not care about other object numbers. Assume partition.
                    raise NotObjectNumberError

            except NotObjectNumberError:
                # assume partition
                try:
                    partition = library.partition(ident)
                    partitions.add(partition.vid)
                    new_parts.append(partition.vid)
                except NotFoundError:
                    # Ok, maybe it is just a normal identifier...
                    new_parts.append(ident)
        else:
            new_parts.append(token)

    return ' '.join(new_parts).strip(), tables, partitions