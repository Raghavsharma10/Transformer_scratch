def validate_body(schema):
    """Validate the body of incoming requests for a flask view.

    An example usage might look like this::

        from snapstore_schemas import validate_body


        @validate_body({
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'snap_id': {'type': 'string'},
                    'series': {'type': 'string'},
                    'name': {'type': 'string'},
                    'title': {'type': 'string'},
                    'keywords': {
                        'type': 'array',
                        'items': {'type': 'string'}
                    },
                    'summary': {'type': 'string'},
                    'description': {'type': 'string'},
                    'created_at': {'type': 'string'},
                },
                'required': ['snap_id', 'series'],
                'additionalProperties': False
            }
        })
        def my_flask_view():
            # view code here
            return "Hello World", 200

    All incoming request that have been routed to this view will be matched
    against the specified schema. If the request body does not match the schema
    an instance of `DataValidationError` will be raised.

    By default this will cause the flask application to return a 500 response,
    but this can be customised by telling flask how to handle these exceptions.
    The exception instance has an 'error_list' attribute that contains a list
    of all the errors encountered while processing the request body.
    """
    location = get_callsite_location()

    def decorator(fn):
        validate_schema(schema)
        wrapper = wrap_request(fn, schema)
        record_schemas(
            fn, wrapper, location, request_schema=sort_schema(schema))
        return wrapper

    return decorator