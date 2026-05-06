def _lookup_model(cls, kind, default_model=None):
    """Get the model class for the kind.

    Args:
      kind: A string representing the name of the kind to lookup.
      default_model: The model class to use if the kind can't be found.

    Returns:
      The model class for the requested kind.
    Raises:
      KindError: The kind was not found and no default_model was provided.
    """
    modelclass = cls._kind_map.get(kind, default_model)
    if modelclass is None:
      raise KindError(
          "No model class found for kind '%s'. Did you forget to import it?" %
          kind)
    return modelclass