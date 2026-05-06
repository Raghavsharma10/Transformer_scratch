def delete(callback=None, path=None, method=Method.DELETE, tags=None, summary="Delete specified resource.",
           middleware=None):
    # type: (Callable, Path, Methods, Tags, str, List[Any]) -> Operation
    """
    Decorator to configure an operation that deletes resource.
    """
    def inner(c):
        op = Operation(c, path or PathParam('{key_field}'), method, None, tags, summary, middleware)
        op.responses.add(Response(HTTPStatus.NO_CONTENT, "{name} has been deleted.", None))
        op.responses.add(Response(HTTPStatus.NOT_FOUND, "Not found", Error))
        return op
    return inner(callback) if callback else inner