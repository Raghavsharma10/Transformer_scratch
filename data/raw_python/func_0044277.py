def disable_all_tokens(platform, user_id, on_error=None, on_success=None):
    """ Disable ALL device tokens for the given user on the specified platform.

    :param str platform The platform which to disable token on. One of either
    Google Cloud Messaging (outbound.GCM) or Apple Push Notification Service
    (outbound.APNS).

    :param str | number user_id: the id you use to identify a user. this should
    be static for the lifetime of a user.

    :param func on_error: An optional function to call in the event of an error.
    on_error callback should take 2 parameters: `code` and `error`. `code` will be
    one of outbound.ERROR_XXXXXX. `error` will be the corresponding message.

    :param func on_success: An optional function to call if/when the API call succeeds.
    on_success callback takes no parameters.
    """
    __device_token(platform, False, user_id, all=True, on_error=on_error, on_success=on_success)