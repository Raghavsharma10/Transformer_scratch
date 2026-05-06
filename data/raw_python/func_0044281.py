def identify(user_id, previous_id=None, group_id=None, group_attributes=None,
            first_name=None, last_name=None, email=None,
            phone_number=None, apns_tokens=None, gcm_tokens=None,
            attributes=None, on_error=None, on_success=None):
    """ Identifying a user creates a record of your user in Outbound. Identify
    calls should be made prior to sending any track events for a user.

    :param str | number user_id: the id you use to identify a user. this should
    be static for the lifetime of a user.

    :param str | number previous_id: OPTIONAL the id you previously used to identify the user.

    :param str | number group_id: OPTIONAL  the id that identifies a group of users the current user
    belongs to.

    :param dict group_attributes: OPTIONAL An optional dictionary of attributes that are shared
    among the group this user belongs to.

    :param str first_name: OPTIONAL the user's first name.

    :param str last_name: OPTIONAL the user's last name.

    :param str email: OPTIONAL the user's email address.

    :param str phone_number: OPTIONAL the user's phone number.

    :param str | list apns_tokens: OPTIONAL the device tokens for the user's iOS
    devices. If a single string is given it is put into a list.

    :param str | list gcm_tokens: OPTIONAL the device tokens for the user's Android
    devices. If a single string is given it is put into a list.

    :param dict attributes: An optional dictionary with any additional freeform
    attributes describing the user.

    :param func on_error: An optional function to call in the event of an error.
    on_error callback should take 2 parameters: `code` and `error`. `code` will be
    one of outbound.ERROR_XXXXXX. `error` will be the corresponding message.

    :param func on_success: An optional function to call if/when the API call succeeds.
    on_success callback takes no parameters.
    """

    on_error = on_error or __on_error
    on_success = on_success or __on_success

    if not __is_init():
        on_error(ERROR_INIT, __error_message(ERROR_INIT))
        return

    if not isinstance(user_id, six.string_types + (Number,)):
        on_error(ERROR_USER_ID, __error_message(ERROR_USER_ID))
        return

    data = __user(
        first_name,
        last_name,
        email,
        phone_number,
        apns_tokens,
        gcm_tokens,
        attributes,
        previous_id,
        group_id,
        group_attributes,)
    data['user_id'] = user_id

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