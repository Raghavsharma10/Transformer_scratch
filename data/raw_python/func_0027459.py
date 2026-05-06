def get_issues_by_resource(resource_id, table):
    """Get all issues for a specific job."""

    v1_utils.verify_existence_and_get(resource_id, table)

    # When retrieving the issues for a job, we actually retrieve
    # the issues attach to the job itself + the issues attached to
    # the components the job has been run with.
    if table.name == 'jobs':
        JJI = models.JOIN_JOBS_ISSUES
        JJC = models.JOIN_JOBS_COMPONENTS
        JCI = models.JOIN_COMPONENTS_ISSUES

        # Get all the issues attach to all the components attach to a job
        j1 = sql.join(
            _TABLE,
            sql.join(
                JCI,
                JJC,
                sql.and_(
                    JCI.c.component_id == JJC.c.component_id,
                    JJC.c.job_id == resource_id,
                ),
            ),
            _TABLE.c.id == JCI.c.issue_id,
        )

        query = sql.select([_TABLE]).select_from(j1)
        rows = flask.g.db_conn.execute(query)
        rows = [dict(row) for row in rows]

        # Get all the issues attach to a job
        j2 = sql.join(
            _TABLE,
            JJI,
            sql.and_(
                _TABLE.c.id == JJI.c.issue_id,
                JJI.c.job_id == resource_id
            )
        )
        query2 = sql.select([_TABLE]).select_from(j2)
        rows2 = flask.g.db_conn.execute(query2)
        rows += [dict(row) for row in rows2]

    # When retrieving the issues for a component, we only retrieve the
    # issues attached to the specified component.
    else:
        JCI = models.JOIN_COMPONENTS_ISSUES

        query = (sql.select([_TABLE])
                 .select_from(JCI.join(_TABLE))
                 .where(JCI.c.component_id == resource_id))

        rows = flask.g.db_conn.execute(query)
        rows = [dict(row) for row in rows]

    for row in rows:
        if row['tracker'] == 'github':
            l_tracker = github.Github(row['url'])
        elif row['tracker'] == 'bugzilla':
            l_tracker = bugzilla.Bugzilla(row['url'])
        row.update(l_tracker.dump())

    return flask.jsonify({'issues': rows,
                          '_meta': {'count': len(rows)}})