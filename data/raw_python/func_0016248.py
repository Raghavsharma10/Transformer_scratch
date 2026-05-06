def get_oauth_authcfg(authcfg_id=AUTHCFG_ID):
    """Check if the given authcfg_id (or the default) exists, and if it's valid
    OAuth2, return the configuration or None"""
    # Handle empty strings
    if not authcfg_id:
        authcfg_id = AUTHCFG_ID
    configs = auth_manager().availableAuthMethodConfigs()
    if authcfg_id in configs \
            and configs[authcfg_id].isValid() \
            and configs[authcfg_id].method() == 'OAuth2':
        return configs[authcfg_id]
    return None