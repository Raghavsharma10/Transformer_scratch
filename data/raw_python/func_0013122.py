def _to_base_type(self, msg):
    """Convert a Message value to a Model instance (entity)."""
    ent = _message_to_entity(msg, self._modelclass)
    ent.blob_ = self._protocol_impl.encode_message(msg)
    return ent