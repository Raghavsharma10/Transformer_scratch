def detail(callback=None, path=None, method=Method.GET, resource=None, tags=None, summary="Get specified resource.",
           middleware=None):
    # type: (Callable, Path, Methods, Resource, Tags, str, List[Any]) -> Operation
    """
    Decorator to configure an operation that fetches a resource.
    """
    def inner(c):
        op = Operation(c, path or PathParam('{key_field}'), method, resource, tags, summary, middleware)
        op.responses.add(Response(HTTPStatus.OK, "Get a {name}"))
        op.responses.add(Response(HTTPStatus.NOT_FOUND, "Not found", Error))
        return op
    return inner(callback) if callback else inner