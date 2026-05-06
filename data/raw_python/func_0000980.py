def validate_output(schema):
    """Validate the body of a response from a flask view.

    Like `validate_body`, this function compares a json document to a
    jsonschema specification. However, this function applies the schema to the
    view response.

    Instead of the view returning a flask response object, it should instead
    return a Python list or dictionary. For example::

        from snapstore_schemas import validate_output

        @validate_output({
            'type': 'object',
            'properties': {
                'ok': {'type': 'boolean'},
            },
            'required': ['ok'],
            'additionalProperties': False
        }
        def my_flask_view():
            # view code here
            return {'ok': True}

    Every view response will be evaluated against the schema. Any that do not
    comply with the schema will cause DataValidationError to be raised.
    """
    location = get_callsite_location()

    def decorator(fn):
        validate_schema(schema)
        wrapper = wrap_response(fn, schema)
        record_schemas(
            fn, wrapper, location, response_schema=sort_schema(schema))
        return wrapper

    return decorator