def web(host, port):
    """Start web application"""
    from .webserver.web import get_app
    get_app().run(host=host, port=port)