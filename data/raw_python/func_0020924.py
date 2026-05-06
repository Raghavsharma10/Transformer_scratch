def after_insert_oai_set(mapper, connection, target):
    """Update records on OAISet insertion."""
    _new_percolator(spec=target.spec, search_pattern=target.search_pattern)
    sleep(2)
    update_affected_records.delay(
        search_pattern=target.search_pattern
    )