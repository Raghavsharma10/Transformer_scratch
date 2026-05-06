def alias(user_id, previous_id, on_error=None, on_success=None):
    """ Alias one user id to another.

    :param str | number user_id: the id you use to identify a user. this will be the user's
    primary user id.

    :param str | number previous_id: the id you previously used to identify a user (or the old user id
    you want to associate with the new user id).

    :param func on_error: An optional function to call in the event of an error.
    on_error callback should take 2 parameters: `code` and `error`. `code` will be
    one of outbound.ERROR_XXXXXX. `error` will be the corresponding message.

    :param func on_success: An optional function to call if/when the API call succeeds.
    on_success callback takes no parameters.
    """
    if not __is_init():
        on_error(ERROR_INIT, __error_message(ERROR_INIT))
        return

    if not isinstance(user_id, six.string_types + (Number,)):
        on_error(ERROR_USER_ID, __error_message(ERROR_USER_ID))
        return

    if not isinstance(previous_id, six.string_types + (Number,)):
        on_error(ERROR_PREVIOUS_ID, __error_message(ERROR_PREVIOUS_ID))
        return

    data = dict(
        user_id=user_id,
        previous_id=previous_id,
    )

    try:
        resp = requests.post(
            "%s/identify" % __BASE_URL,
            data=json.dumps(data),
            headers=__HEADERS,
        )

        if resp.status_code >= 200 and resp.status_code < 400:
            on_success()
        else:
            on_error(ERROR_UNKNOWN, resp.text)
    except requests.exceptions.ConnectionError:
        on_error(ERROR_CONNECTION, __error_message(ERROR_CONNECTION))