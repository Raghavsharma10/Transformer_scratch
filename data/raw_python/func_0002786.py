def login_required(wrapped):
    """
    Requires that the user is logged in and authorized to execute requests
    Except if the method is in authorized_methods of the auth_collection
    Then he can execute the requests even not being authorized
    """
    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        request = args[1]

        auth_collection = settings.AUTH_COLLECTION[
            settings.AUTH_COLLECTION.rfind('.') + 1:
        ].lower()
        auth_document = request.environ.get(auth_collection)

        if auth_document and auth_document.is_authorized(request):
            setattr(request, auth_collection, auth_document)
            return wrapped(*args, **kwargs)

        return Response(response=serialize(UnauthorizedError()), status=401)

    if hasattr(wrapped, 'decorators'):
        wrapper.decorators = wrapped.decorators
        wrapper.decorators.append('login_required')
    else:
        wrapper.decorators = ['login_required']

    return wrapper