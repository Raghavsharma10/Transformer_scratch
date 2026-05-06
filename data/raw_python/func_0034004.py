def gen500(request, baseURI, project=None):
    """Return a 500 error"""
    return HttpResponseServerError(
        render_to_response('plugIt/500.html', {
            'context': {
                'ebuio_baseUrl': baseURI,
                'ebuio_userMode': request.session.get('plugit-standalone-usermode', 'ano'),
            },
            'project': project
        }, context_instance=RequestContext(request)))