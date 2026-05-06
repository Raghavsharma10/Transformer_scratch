def b(field: str, kwargs: Dict[str, Any],
      present: Optional[Any] = None, missing: Any = '') -> str:
    """
    Return `present` value (default to `field`) if `field` in `kwargs` and
    Truthy, otherwise return `missing` value
    """
    if kwargs.get(field):
        return field if present is None else str(present)
    return str(missing)