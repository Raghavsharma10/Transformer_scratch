def api_home(request, key=None, hproPk=None):
    """Show the home page for the API with all methods"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    return render_to_response('plugIt/api.html', {}, context_instance=RequestContext(request))