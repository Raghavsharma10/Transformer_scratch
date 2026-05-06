def jsonresolver_loader(url_map):
    """Resolve the OpenAIRE grant."""
    from flask import current_app
    url_map.add(Rule(
        '/grants/10.13039/<path:doi_grant_code>',
        endpoint=resolve_grant_endpoint,
        host=current_app.config['OPENAIRE_JSONRESOLVER_GRANTS_HOST']))