def preload_pages():
    """Register all pages before the first application request."""
    try:
        _add_url_rule([page.url for page in Page.query.all()])
    except Exception:  # pragma: no cover
        current_app.logger.warn('Pages were not loaded.')
        raise