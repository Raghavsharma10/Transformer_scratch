def after_delete_oai_set(mapper, connection, target):
    """Update records on OAISet deletion."""
    _delete_percolator(spec=target.spec, search_pattern=target.search_pattern)
    sleep(2)
    update_affected_records.delay(
        spec=target.spec
    )