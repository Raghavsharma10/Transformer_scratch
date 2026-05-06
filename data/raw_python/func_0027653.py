def get_to_purge_archived_resources(user, table):
    """List the entries to be purged from the database. """

    if user.is_not_super_admin():
        raise dci_exc.Unauthorized()

    archived_resources = get_archived_resources(table)

    return flask.jsonify({table.name: archived_resources,
                          '_meta': {'count': len(archived_resources)}})