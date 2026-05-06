def _make_response(sijax_response):
    """Takes a Sijax response object and returns a
    valid Flask response object."""
    from types import GeneratorType

    if isinstance(sijax_response, GeneratorType):
        # Streaming response using a generator (non-JSON response).
        # Upon returning a response, Flask would automatically destroy
        # the request data and uploaded files - done by `flask.ctx.RequestContext.auto_pop()`
        # We can't allow that, since the user-provided callback we're executing
        # from within the generator may want to access request data/files.
        # That's why we'll tell Flask to preserve the context and we'll clean up ourselves.

        request.environ['flask._preserve_context'] = True

        # Clean-up code taken from `flask.testing.TestingClient`
        def clean_up_context():
            top = _request_ctx_stack.top
            if top is not None and top.preserved:
                top.pop()

        # As per the WSGI specification, `close()` would be called on iterator responses.
        # Let's wrap the iterator in another one, which will forward that `close()` call to our clean-up callback.
        response = Response(ClosingIterator(sijax_response, clean_up_context), direct_passthrough=True)
    else:
        # Non-streaming response - a single JSON string
        response = Response(sijax_response)

    return response