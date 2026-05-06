def gen403(request, baseURI, reason, project=None):
    """Return a 403 error"""
    orgas = None
    public_ask = False

    if not settings.PIAPI_STANDALONE:
        from organizations.models import Organization

        if project and project.plugItLimitOrgaJoinable:
            orgas = project.plugItOrgaJoinable.order_by('name').all()
        else:
            orgas = Organization.objects.order_by('name').all()

        rorgas = []

        # Find and exclude the visitor orga
        for o in orgas:
            if str(o.pk) == settings.VISITOR_ORGA_PK:
                public_ask = True
            else:
                rorgas.append(o)

        orgas = rorgas

    return HttpResponseForbidden(render_to_response('plugIt/403.html', {'context':
        {
            'reason': reason,
            'orgas': orgas,
            'public_ask': public_ask,
            'ebuio_baseUrl': baseURI,
            'ebuio_userMode': request.session.get('plugit-standalone-usermode', 'ano'),
        },
        'project': project
    }, context_instance=RequestContext(request)))