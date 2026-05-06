def action(callback=None, name=None, path=None, methods=Method.GET, resource=None, tags=None,
           summary=None, middleware=None):
    # type: (Callable, Path, Path, Methods, Type[Resource], Tags, str, List[Any]) -> Operation
    """
    Decorator to apply an action to a resource. An action is applied to a `detail` operation.
    """
    # Generate action path
    path = path or '{key_field}'
    if name:
        path += name

    def inner(c):
        return Operation(c, path, methods, resource, tags, summary, middleware)
    return inner(callback) if callback else inner