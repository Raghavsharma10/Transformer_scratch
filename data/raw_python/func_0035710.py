def dynamic(name: str, expression: Union[type, Callable[[Type[Any]], type]]) \
        -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # TODO type annotations for pass-through decorator
    """
    
    :param name: 
    :param expression: a subclass of ``type`` or a callable in the format ``(owner: Type[Any]) -> type``.
    :return: 
    """

    def decorator(func):
        if not hasattr(func, '__dynamic__'):
            func.__dynamic__ = {name: expression}
        else:
            func.__dynamic__[name] = expression
        return func

    return decorator