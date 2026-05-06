def socket_options(instance):
    """Ensure the keys of the 'options' property of the socket-ext extension of
    network-traffic objects are only valid socket options (SO_*).
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'network-traffic'):
            try:
                options = obj['extensions']['socket-ext']['options']
            except KeyError:
                continue

            for opt in options:
                if opt not in enums.SOCKET_OPTIONS:
                    yield JSONError("The 'options' property of object '%s' "
                                    "contains a key ('%s') that is not a valid"
                                    " socket option (SO_*)."
                                    % (key, opt), instance['id'], 'socket-options')