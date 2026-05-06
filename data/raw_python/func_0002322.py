def _new_render(response):
    """
    Decorator for the TemplateResponse.render() function
    """
    orig_render = response.__class__.render

    # No arguments, is used as bound method.
    def _inner_render():
        try:
            return orig_render(response)
        except HttpRedirectRequest as e:
            return HttpResponseRedirect(e.url, status=e.status)

    return _inner_render