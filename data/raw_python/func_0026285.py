def _get_template_dirs():
    """existing directories where to search for jinja2 templates. The order
    is important. The first found template from the first found dir wins!"""
    return filter(lambda x: os.path.exists(x), [
        # user dir
        os.path.join(os.path.expanduser('~'), '.py2pack', 'templates'),
        # system wide dir
        os.path.join('/', 'usr', 'share', 'py2pack', 'templates'),
        # usually inside the site-packages dir
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
    ])