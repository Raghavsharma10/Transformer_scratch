def _records_commit(record_ids):
    """Commit all records."""
    for record_id in record_ids:
        record = Record.get_record(record_id)
        record.commit()