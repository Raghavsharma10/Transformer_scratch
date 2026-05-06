def resp2flask(resp):
    """Convert an oic.utils.http_util instance to Flask."""
    if isinstance(resp, Redirect) or isinstance(resp, SeeOther):
        code = int(resp.status.split()[0])
        raise cherrypy.HTTPRedirect(resp.message, code)
    return resp.message, resp.status, resp.headers