def unattach_issue(resource_id, issue_id, table):
    """Unattach an issue from a specific job."""

    v1_utils.verify_existence_and_get(issue_id, _TABLE)
    if table.name == 'jobs':
        join_table = models.JOIN_JOBS_ISSUES
        where_clause = sql.and_(join_table.c.job_id == resource_id,
                                join_table.c.issue_id == issue_id)
    else:
        join_table = models.JOIN_COMPONENTS_ISSUES
        where_clause = sql.and_(join_table.c.component_id == resource_id,
                                join_table.c.issue_id == issue_id)

    query = join_table.delete().where(where_clause)
    result = flask.g.db_conn.execute(query)

    if not result.rowcount:
        raise dci_exc.DCIConflict('%s_issues' % table.name, issue_id)

    return flask.Response(None, 204, content_type='application/json')