def _commonPrefetchDeclarativeIds(engine, mutex,
                                  Declarative, count) -> Optional[Iterable[int]]:
    """ Common Prefetch Declarative IDs

    This function is used by the worker and server
    """
    if not count:
        logger.debug("Count was zero, no range returned")
        return

    conn = engine.connect()
    transaction = conn.begin()
    mutex.acquire()
    try:
        sequence = Sequence('%s_id_seq' % Declarative.__tablename__,
                            schema=Declarative.metadata.schema)

        if isPostGreSQLDialect(engine):
            sql = "SELECT setval('%(seq)s', (select nextval('%(seq)s') + %(add)s), true)"
            sql %= {
                'seq': '"%s"."%s"' % (sequence.schema, sequence.name),
                'add': count
            }
            nextStartId = conn.execute(sql).fetchone()[0]
            startId = nextStartId - count

        elif isMssqlDialect(engine):
            startId = conn.execute(
                'SELECT NEXT VALUE FOR "%s"."%s"'
                % (sequence.schema, sequence.name)
            ).fetchone()[0] + 1

            nextStartId = startId + count

            conn.execute('alter sequence "%s"."%s" restart with %s'
                         % (sequence.schema, sequence.name, nextStartId))

        else:
            raise NotImplementedError()

        transaction.commit()

        return iter(range(startId, nextStartId))

    finally:
        mutex.release()
        conn.close()