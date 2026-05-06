def main(request, query, hproPk=None, returnMenuOnly=False):
    """ Main method called for main page"""
    if settings.PIAPI_STANDALONE:
        global plugIt, baseURI

        # Check if settings are ok
        if settings.PIAPI_ORGAMODE and settings.PIAPI_REALUSERS:
            return gen404(request, baseURI,
                          "Configuration error. PIAPI_ORGAMODE and PIAPI_REALUSERS both set to True !")

        hproject = None
    else:
        (plugIt, baseURI, hproject) = getPlugItObject(hproPk)
        hproject.update_last_access(request.user)

    # Check for SSL Requirements and redirect to ssl if necessary
    if hproject and hproject.plugItRequiresSsl:
        if not request.is_secure():
            secure_url = 'https://{0}{1}'.format(request.get_host(), request.get_full_path())
            return HttpResponsePermanentRedirect(secure_url)

    orgaMode = None
    currentOrga = None
    plugItMenuAction = None
    availableOrga = []

    # If standalone mode, change the current user and orga mode based on parameters
    if settings.PIAPI_STANDALONE:

        if not settings.PIAPI_REALUSERS:
            currentUserMode = request.session.get('plugit-standalone-usermode', 'ano')

            request.user = generate_user(mode=currentUserMode)

            orgaMode = settings.PIAPI_ORGAMODE

            currentOrga = SimpleOrga()
            currentOrga.name = request.session.get('plugit-standalone-organame', 'EBU')
            currentOrga.pk = request.session.get('plugit-standalone-orgapk', '-1')
            currentOrga.ebu_codops = request.session.get('plugit-standalone-orgacodops', 'zzebu')
        else:
            request.user.ebuio_member = request.user.is_staff
            request.user.ebuio_admin = request.user.is_superuser
            request.user.subscription_labels = _get_subscription_labels(request.user, hproject)

        proxyMode = settings.PIAPI_PROXYMODE
        plugItMenuAction = settings.PIAPI_PLUGITMENUACTION

        # TODO Add STANDALONE Orgas here
        # availableOrga.append((orga, isAdmin))

        # Get meta, if not in proxy mode
        if not proxyMode:

            try:
                meta = plugIt.getMeta(query)
            except Exception as e:
                report_backend_error(request, e, 'meta', hproPk)
                meta = None

            if not meta:
                return gen404(request, baseURI, 'meta')
        else:
            meta = {}

    else:
        request.user.ebuio_member = hproject.isMemberRead(request.user)
        request.user.ebuio_admin = hproject.isMemberWrite(request.user)
        request.user.subscription_labels = _get_subscription_labels(request.user, hproject)
        orgaMode = hproject.plugItOrgaMode
        proxyMode = hproject.plugItProxyMode
        plugItMenuAction = hproject.plugItMenuAction

        # Get meta, if not in proxy mode
        if not proxyMode:

            try:
                meta = plugIt.getMeta(query)
            except Exception as e:
                report_backend_error(request, e, 'meta', hproPk)
                meta = None

            if not meta:
                return gen404(request, baseURI, 'meta')
        else:
            meta = {}

        if orgaMode:
            # List available orgas
            if request.user.is_authenticated():
                # If orga limited only output the necessary orgas to which the user has access
                if hproject and hproject.plugItLimitOrgaJoinable:
                    # Get List of Plugit Available Orgas first
                    projectOrgaIds = hproject.plugItOrgaJoinable.order_by('name').values_list('pk', flat=True)
                    for (orga, isAdmin) in request.user.getOrgas(distinct=True):
                        if orga.pk in projectOrgaIds:
                            availableOrga.append((orga, isAdmin))

                else:
                    availableOrga = request.user.getOrgas(distinct=True)

            if not availableOrga:
                # TODO HERE TO CHANGE PUBLIC
                if not meta.get('public'):  # Page is not public, raise 403
                    return gen403(request, baseURI, 'no_orga_in_orgamode', hproject)
            else:
                # Build the current orga
                realCurrentOrga = get_current_orga(request, hproject, availableOrga)

                currentOrga = SimpleOrga()

                currentOrga.pk = realCurrentOrga.pk
                currentOrga.name = realCurrentOrga.name
                currentOrga.ebu_codops = realCurrentOrga.ebu_codops

                # Get rights
                request.user.ebuio_orga_member = realCurrentOrga.isMember(request.user)
                request.user.ebuio_orga_admin = realCurrentOrga.isOwner(request.user)

    cacheKey = get_cache_key(request, meta, orgaMode, currentOrga)

    # Check access rights
    error = check_rights_and_access(request, meta, hproject)

    if error:
        return error

    # Check cache
    (cache, menucache, context) = find_in_cache(cacheKey)

    if cache:
        return build_final_response(request, meta, cache, menucache, hproject, proxyMode, context)

    # Build parameters
    getParameters, postParameters, files = build_parameters(request, meta, orgaMode, currentOrga)

    # Bonus headers
    things_to_add = build_extra_headers(request, proxyMode, orgaMode, currentOrga)

    current_session = get_current_session(request, hproPk)

    # Do the action
    try:
        (data, session_to_set, headers_to_set) = plugIt.doAction(query, request.method, getParameters, postParameters, files, things_to_add, proxyMode=proxyMode, session=current_session)
    except Exception as e:
        report_backend_error(request, e, 'meta', hproPk)
        return gen500(request, baseURI)

    update_session(request, session_to_set, hproPk)

    # Handle special case (redirect, etc..)
    spe_cases = handle_special_cases(request, data, baseURI, meta)
    if spe_cases:

        for header, value in headers_to_set.items():
            spe_cases[header] = value

        return spe_cases

    # Save data for proxyMode
    if proxyMode:
        rendered_data = data
        data = {}
    else:
        rendered_data = None

    # Get template
    (templateContent, templateError) = get_template(request, query, meta, proxyMode)

    if templateError:
        return templateError

    # Build the context
    context = build_context(request, data, hproject, orgaMode, currentOrga, availableOrga)

    # Render the result
    menu = None  # Some page may not have a menu
    (result, menu) = render_data(context, templateContent, proxyMode, rendered_data, plugItMenuAction)

    # Cache the result for future uses if requested
    cache_if_needed(cacheKey, result, menu, context, meta)

    # Return menu only : )
    if returnMenuOnly:
        return menu

    # Return the final response
    final = build_final_response(request, meta, result, menu, hproject, proxyMode, context)

    for header, value in headers_to_set.items():
        final[header] = value

    return final