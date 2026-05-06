def check(func):
    """ 
    Check the permissions, http method and login state.
    """
    def iCheck(request, *args, **kwargs):
        if not request.method == "POST":
            return HttpResponseBadRequest("Must be POST request.")
        follow = func(request, *args, **kwargs)
        if request.is_ajax():
            return HttpResponse('ok')
        try:
            if 'next' in request.GET:
                return HttpResponseRedirect(request.GET.get('next'))
            if 'next' in request.POST:
                return HttpResponseRedirect(request.POST.get('next'))
            return HttpResponseRedirect(follow.target.get_absolute_url())
        except (AttributeError, TypeError):
            if 'HTTP_REFERER' in request.META:
                return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
            if follow:
                return HttpResponseServerError('"%s" object of type ``%s`` has no method ``get_absolute_url()``.' % (
                    unicode(follow.target), follow.target.__class__))
            return HttpResponseServerError('No follow object and `next` parameter found.')
    return iCheck