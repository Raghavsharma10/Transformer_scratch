def auth():
    """
        Create the folder/directories for an Auth service.
    """
    # the template context
    context = {
        'name': 'auth',
    }

    render_template(template='common', context=context)
    render_template(template='auth', context=context)