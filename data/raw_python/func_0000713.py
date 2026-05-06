def connect_handler(cv, provider):
    """Shared method to handle the connection process

    :param connection_values: A dictionary containing the connection values
    :param provider_id: The provider ID the connection shoudl be made to
    """
    cv.setdefault('user_id', current_user.get_id())
    connection = _datastore.find_connection(
        provider_id=cv['provider_id'], provider_user_id=cv['provider_user_id'])

    if connection is None:
        after_this_request(_commit)
        connection = _datastore.create_connection(**cv)
        msg = ('Connection established to %s' % provider.name, 'success')
        connection_created.send(current_app._get_current_object(),
                                user=current_user._get_current_object(),
                                connection=connection)
    else:
        msg = ('A connection is already established with %s '
               'to your account' % provider.name, 'notice')
        connection_failed.send(current_app._get_current_object(),
                               user=current_user._get_current_object())

    redirect_url = session.pop(config_value('POST_OAUTH_CONNECT_SESSION_KEY'),
                               get_url(config_value('CONNECT_ALLOW_VIEW')))

    do_flash(*msg)
    return redirect(redirect_url)