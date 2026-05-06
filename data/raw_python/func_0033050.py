def minter(record_uuid, data, pid_type, key):
    """Mint PIDs for a record."""
    pid = PersistentIdentifier.create(
        pid_type,
        data[key],
        object_type='rec',
        object_uuid=record_uuid,
        status=PIDStatus.REGISTERED
    )
    for scheme, identifier in data['identifiers'].items():
        if identifier:
            PersistentIdentifier.create(
                scheme,
                identifier,
                object_type='rec',
                object_uuid=record_uuid,
                status=PIDStatus.REGISTERED
            )
    return pid