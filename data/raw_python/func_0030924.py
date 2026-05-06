def ensure_json_content_type(req, app):
    """
    ErrorMiddleware returns hard-coded content-type text/html.
    Here we force it to be application/json.
    """
    res = req.get_response(app, catch_exc_info=True)
    res.content_type = 'application/json; charset=utf-8'
    return res