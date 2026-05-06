def get_jobs_events_from_sequence(user, sequence):
    """Get all the jobs events from a given sequence number."""

    args = schemas.args(flask.request.args.to_dict())

    if user.is_not_super_admin():
        raise dci_exc.Unauthorized()

    query = sql.select([models.JOBS_EVENTS]). \
        select_from(models.JOBS_EVENTS.join(models.JOBS,
                    models.JOBS.c.id == models.JOBS_EVENTS.c.job_id)). \
        where(_TABLE.c.id >= sequence)
    sort_list = v1_utils.sort_query(args['sort'], _JOBS_EVENTS_COLUMNS,
                                    default='created_at')
    query = v1_utils.add_sort_to_query(query, sort_list)

    if args.get('limit', None):
        query = query.limit(args.get('limit'))
    if args.get('offset', None):
        query = query.offset(args.get('offset'))

    rows = flask.g.db_conn.execute(query).fetchall()

    query_nb_rows = sql.select([func.count(models.JOBS_EVENTS.c.id)])
    nb_rows = flask.g.db_conn.execute(query_nb_rows).scalar()

    return json.jsonify({'jobs_events': rows, '_meta': {'count': nb_rows}})