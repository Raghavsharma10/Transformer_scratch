def create(callback=None, path=None, method=Method.POST, resource=None, tags=None, summary="Create a new resource",
           middleware=None):
    # type: (Callable, Path, Methods, Resource, Tags, str, List[Any]) -> Operation
    """
    Decorator to configure an operation that creates a resource.
    """
    def inner(c):
        op = ResourceOperation(c, path or NoPath, method, resource, tags, summary, middleware)
        op.responses.add(Response(HTTPStatus.CREATED, "{name} has been created"))
        op.responses.add(Response(HTTPStatus.BAD_REQUEST, "Validation failed.", Error))
        return op
    return inner(callback) if callback else inner