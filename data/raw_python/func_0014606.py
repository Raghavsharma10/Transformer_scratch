def check_csrf_token():
    """Checks that token is correct, aborting if not"""
    if request.method in ("GET",): # not exhaustive list
        return
    token = request.form.get("csrf_token")
    if token is None:
        app.logger.warning("Expected CSRF Token: not present")
        abort(400)
    if not safe_str_cmp(token, csrf_token()):
        app.logger.warning("CSRF Token incorrect")
        abort(400)