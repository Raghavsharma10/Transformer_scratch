def csw_global_dispatch(request, url=None, catalog_id=None):
    """pycsw wrapper"""

    if request.user.is_authenticated():  # turn on CSW-T
        settings.REGISTRY_PYCSW['manager']['transactions'] = 'true'

    env = request.META.copy()

    # TODO: remove this workaround
    # HH should be able to pass env['wsgi.input'] without hanging
    # details at https://github.com/cga-harvard/HHypermap/issues/94
    if request.method == 'POST':
        from StringIO import StringIO
        env['wsgi.input'] = StringIO(request.body)

    env.update({'local.app_root': os.path.dirname(__file__),
                'REQUEST_URI': request.build_absolute_uri()})

    # if this is a catalog based CSW, then update settings
    if url is not None:
        settings.REGISTRY_PYCSW['server']['url'] = url
    if catalog_id is not None:
        settings.REGISTRY_PYCSW['repository']['filter'] = 'catalog_id = %d' % catalog_id

    csw = server.Csw(settings.REGISTRY_PYCSW, env)

    content = csw.dispatch_wsgi()

    # pycsw 2.0 has an API break:
    # pycsw < 2.0: content = xml_response
    # pycsw >= 2.0: content = [http_status_code, content]
    # deal with the API break

    if isinstance(content, list):  # pycsw 2.0+
        content = content[1]

    response = HttpResponse(content, content_type=csw.contenttype)

    # TODO: Fix before 1.0 release. CORS should not be enabled blindly like this.
    response['Access-Control-Allow-Origin'] = '*'
    return response