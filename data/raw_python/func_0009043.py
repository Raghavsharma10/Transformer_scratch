def set_uuid(obj, uuid=None):
    """Set the uuid attribute of an HDF5 object. Use this method to ensure correct dtype """
    from uuid import uuid4, UUID
    if uuid is None:
        uuid = uuid4()
    elif isinstance(uuid, bytes):
        if len(uuid) == 16:
            uuid = UUID(bytes=uuid)
        else:
            uuid = UUID(hex=uuid)

    if "uuid" in obj.attrs:
        del obj.attrs["uuid"]
    obj.attrs.create("uuid", str(uuid).encode('ascii'), dtype="|S36")