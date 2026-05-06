def record_schemas(
        fn, wrapper, location, request_schema=None, response_schema=None):
    """Support extracting the schema from the decorated function."""
    # have we already been decorated by an acceptable api call?
    has_acceptable = hasattr(fn, '_acceptable_metadata')

    if request_schema is not None:
        # preserve schema for later use
        wrapper._request_schema = wrapper._request_schema = request_schema
        wrapper._request_schema_location = location
        if has_acceptable:
            fn._acceptable_metadata._request_schema = request_schema
            fn._acceptable_metadata._request_schema_location = location

    if response_schema is not None:
        # preserve schema for later use
        wrapper._response_schema = wrapper._response_schema = response_schema
        wrapper._response_schema_location = location
        if has_acceptable:
            fn._acceptable_metadata._response_schema = response_schema
            fn._acceptable_metadata._response_schema_location = location