def gen404(request, baseURI, reason, project=None):
    """Return a 404 error"""
    return HttpResponseNotFound(
        render_to_response('plugIt/404.html', {'context':
            {
                'reason': reason,
                'ebuio_baseUrl': baseURI,
                'ebuio_userMode': request.session.get('plugit-standalone-usermode', 'ano'),
            },
            'project': project
        }, context_instance=RequestContext(request)))