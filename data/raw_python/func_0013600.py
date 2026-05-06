def render_key(app, key=""):
    """
    Renders a view from the app and a key that lets the current session grab
    its token.
    :param app:
    :param key:
    :return: Rendered view
    """
    return app.jinja_env.from_string(KEY_HTML).render(
        key=key)