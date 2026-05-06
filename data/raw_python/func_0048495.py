def response(status, description, resource=DefaultResource):
    # type: (HTTPStatus, str, Optional[Resource]) -> Callable
    """
    Define an expected response.

    The values are based off `Swagger <https://swagger.io/specification>`_.

    """
    def inner(o):
        value = Response(status, description, resource)
        try:
            getattr(o, 'responses').add(value)
        except AttributeError:
            setattr(o, 'responses', {value})
        return o
    return inner