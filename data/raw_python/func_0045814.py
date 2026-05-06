def listing(callback=None, path=None, method=Method.GET, resource=None, tags=None, summary="List resources",
            middleware=None, default_limit=50, max_limit=None, use_wrapper=True):
    # type: (Callable, Path, Methods, Resource, Tags, str, List[Any], int, int) -> Operation
    """
    Decorator to configure an operation that returns a list of resources.
    """
    op_type = WrappedListOperation if use_wrapper else ListOperation

    def inner(c):
        op = op_type(c, path or NoPath, method, resource, tags, summary, middleware,
                     default_limit=default_limit, max_limit=max_limit)
        op.responses.add(Response(HTTPStatus.OK, "Listing of resources", Listing))
        return op
    return inner(callback) if callback else inner