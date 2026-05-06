def create_or_update_record(data, pid_type, id_key, minter):
    """Register a funder or grant."""
    resolver = Resolver(
        pid_type=pid_type, object_type='rec', getter=Record.get_record)

    try:
        pid, record = resolver.resolve(data[id_key])
        data_c = deepcopy(data)
        del data_c['remote_modified']
        record_c = deepcopy(record)
        del record_c['remote_modified']
        # All grants on OpenAIRE are modified periodically even if nothing
        # has changed. We need to check for actual differences in the metadata
        if data_c != record_c:
            record.update(data)
            record.commit()
            record_id = record.id
            db.session.commit()
            RecordIndexer().index_by_id(str(record_id))
    except PIDDoesNotExistError:
        record = Record.create(data)
        record_id = record.id
        minter(record.id, data)
        db.session.commit()
        RecordIndexer().index_by_id(str(record_id))