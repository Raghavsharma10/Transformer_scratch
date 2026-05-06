def next_sequence_id(session, sequence_ids, parent_vid, table_class, force_query = False):
    """
    Return the next sequence id for a object, identified by the vid of the parent object, and the database prefix
    for the child object. On the first call, will load the max sequence number
    from the database, but subsequence calls will run in process, so this isn't suitable for
    multi-process operation -- all of the tables in a dataset should be created by one process

    The child table must have a sequence_id value.

    :param session: Database session or connection ( must have an execute() method )
    :param sequence_ids: A dict for caching sequence ids
    :param parent_vid: The VID of the parent object, which sets the namespace for the sequence
    :param table_class: Table class of the child object, the one getting a number
    :return:
    """

    from sqlalchemy import text

    seq_col = table_class.sequence_id.property.columns[0].name

    try:
        parent_col = table_class._parent_col
    except AttributeError:
        parent_col = table_class.d_vid.property.columns[0].name

    assert bool(parent_vid)

    key = (parent_vid, table_class.__name__)

    number = sequence_ids.get(key, None)

    if (not number and session) or force_query:

        sql = text("SELECT max({seq_col})+1 FROM {table} WHERE {parent_col} = '{vid}'"
                   .format(table=table_class.__tablename__, parent_col=parent_col,
                           seq_col=seq_col, vid=parent_vid))

        max_id, = session.execute(sql).fetchone()

        if not max_id:
            max_id = 1

        sequence_ids[key] = int(max_id)

    elif not session:
        # There was no session set. This should only happen when the parent object is new, and therefore,
        # there are no child number, so the appropriate starting number is 1. If the object is not new,
        # there will be conflicts.
        sequence_ids[key] = 1

    else:
        # There were no previous numbers, so start with 1
        sequence_ids[key] += 1

    return sequence_ids[key]