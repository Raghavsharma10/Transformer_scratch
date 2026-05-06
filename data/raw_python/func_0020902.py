def update_affected_records(spec=None, search_pattern=None):
    """Update all affected records by OAISet change.

    :param spec: The record spec.
    :param search_pattern: The search pattern.
    """
    chunk_size = current_app.config['OAISERVER_CELERY_TASK_CHUNK_SIZE']
    record_ids = get_affected_records(spec=spec, search_pattern=search_pattern)

    group(
        update_records_sets.s(list(filter(None, chunk)))
        for chunk in zip_longest(*[iter(record_ids)] * chunk_size)
    )()