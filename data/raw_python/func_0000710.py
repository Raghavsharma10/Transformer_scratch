def connect(provider_id):
    """Starts the provider connection OAuth flow"""
    provider = get_provider_or_404(provider_id)
    callback_url = get_authorize_callback('connect', provider_id)
    allow_view = get_url(config_value('CONNECT_ALLOW_VIEW'))
    pc = request.form.get('next', allow_view)
    session[config_value('POST_OAUTH_CONNECT_SESSION_KEY')] = pc
    return provider.authorize(callback_url)