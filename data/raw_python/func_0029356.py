def authenticated_userid(request):
    """Helper function that can be used in ``db_key`` to support `self`
    as a collection key.
    """
    user = getattr(request, 'user', None)
    key = user.pk_field()
    return getattr(user, key)