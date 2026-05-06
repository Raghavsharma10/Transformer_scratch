def csrf_token():
    """
    Generate a token string from bytes arrays. The token in the session is user
    specific.
    """
    if "_csrf_token" not in session:
        session["_csrf_token"] = os.urandom(128)
    return hmac.new(app.secret_key, session["_csrf_token"],
            digestmod=sha1).hexdigest()