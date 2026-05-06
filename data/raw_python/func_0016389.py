def f(field: str, kwargs: Dict[str, Any],
      default: Optional[Any] = None) -> str:
    """ Alias for more readable command construction """
    if default is not None:
        return str(kwargs.get(field, default))
    return str(kwargs[field])