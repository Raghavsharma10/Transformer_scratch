def update(callback=None, path=None, method=Method.PUT, resource=None, tags=None, summary="Update specified resource.",
           middleware=None):
    # type: (Callable, Path, Methods, Resource, Tags, str, List[Any]) -> Operation
    """
    Decorator to configure an operation that updates a resource.
    """
    def inner(c):
        op = ResourceOperation(c, path or PathParam('{key_field}'), method, resource, tags, summary, middleware)
        op.responses.add(Response(HTTPStatus.NO_CONTENT, "{name} has been updated."))
        op.responses.add(Response(HTTPStatus.BAD_REQUEST, "Validation failed.", Error))
        op.responses.add(Response(HTTPStatus.NOT_FOUND, "Not found", Error))
        return op
    return inner(callback) if callback else inner