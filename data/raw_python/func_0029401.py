def get_tunneling(handler, registry):
    """ Allows all methods to be tunneled via GET for dev/debuging
    purposes.
    """
    log.info('get_tunneling enabled')

    def get_tunneling(request):
        if request.method == 'GET':
            method = request.GET.pop('_m', 'GET')
            request.method = method

            if method in ['POST', 'PUT', 'PATCH']:
                get_params = request.GET.mixed()
                valid_params = drop_reserved_params(get_params)
                request.body = six.b(json.dumps(valid_params))
                request.content_type = 'application/json'
                request._tunneled_get = True

        return handler(request)

    return get_tunneling