def build_user_requested_parameters(request, meta):
    """Build the list of parameters requested by the plugit server"""

    postParameters = {}
    getParameters = {}
    files = {}

    # Add parameters requested by the server
    if 'user_info' in meta:
        for prop in meta['user_info']:

            # Test if the value exist, otherwise return None
            value = None
            if hasattr(request.user, prop) and prop in settings.PIAPI_USERDATA:
                value = getattr(request.user, prop)
            else:
                raise Exception('requested user attribute "%s", '
                                'does not exist or requesting is not allowed' % prop)

            # Add informations to get or post parameters, depending on the current method
            if request.method == 'POST':
                postParameters['ebuio_u_' + prop] = value
            else:
                getParameters['ebuio_u_' + prop] = value

    return (getParameters, postParameters, files)