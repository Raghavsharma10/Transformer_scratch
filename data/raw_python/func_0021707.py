def serialize_tag(tag, *, indent=None, compact=False, quote=None):
    """Serialize an nbt tag to its literal representation."""
    serializer = Serializer(indent=indent, compact=compact, quote=quote)
    return serializer.serialize(tag)