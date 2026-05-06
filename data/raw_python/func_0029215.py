def cache_request_user(user_cls, request, user_id):
    """ Helper function to cache currently logged in user.

    User is cached at `request._user`. Caching happens only only
    if user is not already cached or if cached user's pk does not
    match `user_id`.

    :param user_cls: User model class to use for user lookup.
    :param request: Pyramid Request instance.
    :user_id: Current user primary key field value.
    """
    pk_field = user_cls.pk_field()
    user = getattr(request, '_user', None)
    if user is None or getattr(user, pk_field, None) != user_id:
        request._user = user_cls.get_item(**{pk_field: user_id})