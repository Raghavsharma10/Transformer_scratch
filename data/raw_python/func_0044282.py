def track(user_id, event, first_name=None, last_name=None, email=None,
        phone_number=None, apns_tokens=None, gcm_tokens=None,
        user_attributes=None, properties=None, on_error=None, on_success=None, timestamp=None):
    """ For any event you want to track, when a user triggers that event you
    would call this function.

    You can do an identify and track call simultaneously by including all the
    identifiable user information in the track call.

    :param str | number user_id: the id you user who triggered the event.

    :param str first_name: OPTIONAL the user's first name.

    :param str last_name: OPTIONAL the user's last name.

    :param str email: OPTIONAL the user's email address.

    :param str phone_number: OPTIONAL the user's phone number.

    :param str | list apns_tokens: OPTIONAL the device tokens for the user's iOS
    devices. If a single string is given it is put into a list.

    :param str | list gcm_tokens: OPTIONAL the device tokens for the user's Android
    devices. If a single string is given it is put into a list.

    :param dict user_attributes: An optional dictionary with any additional
    freeform attributes describing the user.

    :param dict properties: An optional dictionary with any properties that
    describe the event being track. Example: if the event were "added item to
    cart", you might include a properties named "item" that is the name
    of the item added to the cart.

    :param func on_error: An optional function to call in the event of an error.
    on_error callback should take 1 parameter which will be the error message.

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
    if not isinstance(event, six.string_types):
        on_error(ERROR_EVENT_NAME, __error_message(ERROR_EVENT_NAME))
        return

    data = dict(user_id=user_id, event=event)
    user = __user(
        first_name,
        last_name,
        email,
        phone_number,
        apns_tokens,
        gcm_tokens,
        user_attributes,
        None, None, None)
    if user:
        data['user'] = user

    if properties:
        if isinstance(properties, dict):
            if len(properties) > 0:
                data['properties'] = properties
        else:
            sys.stderr.write('Invalid event properties given. Expected dictionary. ' +
                        'Got %s' % type(properties).__name__)

    if timestamp:
        data['timestamp'] = timestamp
    else:
        data['timestamp'] = int(time.time())

    try:
        resp = requests.post(
            "%s/track" % __BASE_URL,
            data=json.dumps(data),
            headers=__HEADERS,
        )

        if resp.status_code >= 200 and resp.status_code < 400:
            on_success()
        else:
            on_error(ERROR_UNKNOWN, resp.text)
    except requests.exceptions.ConnectionError:
        on_error(ERROR_CONNECTION, __error_message(ERROR_CONNECTION))