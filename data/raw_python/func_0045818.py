def patch(callback=None, path=None, method=Method.PATCH, resource=None, tags=None, summary="Patch specified resource.",
          middleware=None):
    # type: (Callable, Path, Methods, Resource, Tags, str, List[Any]) -> Operation
    """
    Decorator to configure an operation that patches a resource.
    """
    def inner(c):
        op = ResourceOperation(c, path or PathParam('{key_field}'), method, resource, tags, summary, middleware,
                               full_clean=False, default_to_not_supplied=True)
        op.responses.add(Response(HTTPStatus.OK, "{name} has been patched."))
        op.responses.add(Response(HTTPStatus.BAD_REQUEST, "Validation failed.", Error))
        op.responses.add(Response(HTTPStatus.NOT_FOUND, "Not found", Error))
        return op
    return inner(callback) if callback else inner