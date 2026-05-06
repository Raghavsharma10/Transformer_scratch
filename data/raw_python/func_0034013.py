def build_extra_headers(request, proxyMode, orgaMode, currentOrga):
    """Build the list of extra headers"""

    things_to_add = {}

    # If in proxymode, add needed infos to headers
    if proxyMode:

        # User
        for prop in settings.PIAPI_USERDATA:
            if hasattr(request.user, prop):
                things_to_add['user_' + prop] = getattr(request.user, prop)

        # Orga
        if orgaMode:
            things_to_add['orga_pk'] = currentOrga.pk
            things_to_add['orga_name'] = currentOrga.name
            things_to_add['orga_codops'] = currentOrga.ebu_codops

        # General
        things_to_add['base_url'] = baseURI

    if request and hasattr(request, 'META'):
        if 'REMOTE_ADDR' in request.META:
            things_to_add['remote-addr'] = request.META['REMOTE_ADDR']

        if 'HTTP_X_FORWARDED_FOR' in request.META and getattr(settings, 'HONOR_X_FORWARDED_FOR'):
            things_to_add['remote-addr'] = request.META['HTTP_X_FORWARDED_FOR']

        for meta_header, dest_header in [('HTTP_IF_NONE_MATCH', 'If-None-Match'), ('HTTP_ORIGIN', 'Origin'), ('HTTP_ACCESS_CONtROL_REQUEST_METHOD', 'Access-Control-Request-Method'), ('HTTP_ACCESS_CONTROL_REQUEST_HEADERS', 'Access-Control-Request-Headers')]:
            if meta_header in request.META:
                things_to_add[dest_header] = request.META[meta_header]

    return things_to_add