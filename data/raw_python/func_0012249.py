def api():
    """
        Create the folder/directories for an ApiGateway service.
    """
    # the template context
    context = {
        'name': 'api',
        'secret_key': random_string(32)
    }

    render_template(template='common', context=context)
    render_template(template='api', context=context)