def _webfinger(provider, request, **kwargs):
    """Handle webfinger requests."""
    params = urlparse.parse_qs(request)
    if params["rel"][0] == OIC_ISSUER:
        wf = WebFinger()
        return Response(wf.response(params["resource"][0], provider.baseurl),
                        headers=[("Content-Type", "application/jrd+json")])
    else:
        return BadRequest("Incorrect webfinger.")