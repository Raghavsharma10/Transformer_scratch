def key_for_entity_group(cls, key):
    """Return the key for the entity group containing key.

    Args:
      key: a key for an entity group whose __entity_group__ key you want.

    Returns:
      The __entity_group__ key for the entity group containing key.
    """
    return model.Key(cls.KIND_NAME, cls.ID, parent=key.root())