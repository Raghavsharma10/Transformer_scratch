def _add_url_rule(url_or_urls):
    """Register URL rule to application URL map."""
    old = current_app._got_first_request
    # This is bit of cheating to overcome @flask.app.setupmethod decorator.
    current_app._got_first_request = False
    if isinstance(url_or_urls, six.string_types):
        url_or_urls = [url_or_urls]
    map(lambda url: current_app.add_url_rule(url, 'invenio_pages.view', view),
        url_or_urls)
    current_app._got_first_request = old