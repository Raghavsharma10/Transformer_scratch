def filter_validate_response(get_response, params):
    """
    This filter process the returned response. It does 2 things:

    - If the response is a ``sanic.response.HTTPResponse`` and not a
      :class:`rafter.http.Response`, return it immediately.
    - It processes, validates and serializes this response when a schema
      is provided.

    That means that you can always return a normal Sanic's HTTPResponse
    and thus, bypass the validation process when you need to do so.

    .. important::
        The response validation is only effective when:

        - A ``response_schema`` has been provided by the resource definition
        - The resource returns a :class:`rafter.http.Response` instance
          or arbitrary data.
    """

    schema = params.get('response_schema')

    async def decorated_filter(request, *args, **kwargs):
        response = await get_response(request, *args, **kwargs)

        if isinstance(response, HTTPResponse) and \
                not isinstance(response, Response):
            return response

        if not isinstance(response, Response):
            raise TypeError('response is not an instance '
                            'of rafter.http.Response.')

        if schema:
            data = {
                'body': response.data,
                'headers': response.headers
            }

            try:
                model = schema(data, strict=False, validate=False)
                model.validate()
                result = model.to_primitive()
                response.body = result.get('body', None)
                response.headers.update(result.get('headers', {}))
            except BaseError as e:
                log.exception(e)
                abort(500, 'Wrong data output')

        return response

    return decorated_filter