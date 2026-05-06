def setup_oauth(username, password, basemaps_token_uri, authcfg_id=AUTHCFG_ID, authcfg_name=AUTHCFG_NAME):
    """Setup oauth configuration to access the BCS API,
    return authcfg_id on success, None on failure
    """
    cfgjson = {
     "accessMethod" : 0,
     "apiKey" : "",
     "clientId" : "",
     "clientSecret" : "",
     "configType" : 1,
     "grantFlow" : 2,
     "password" : password,
     "persistToken" : False,
     "redirectPort" : '7070',
     "redirectUrl" : "",
     "refreshTokenUrl" : "",
     "requestTimeout" : '30',
     "requestUrl" : "",
     "scope" : "",
     "state" : "",
     "tokenUrl" : basemaps_token_uri,
     "username" : username,
     "version" : 1
    }

    if authcfg_id not in auth_manager().availableAuthMethodConfigs():
        authConfig = QgsAuthMethodConfig('OAuth2')
        authConfig.setId(authcfg_id)
        authConfig.setName(authcfg_name)
        authConfig.setConfig('oauth2config', json.dumps(cfgjson))
        if auth_manager().storeAuthenticationConfig(authConfig):
            return authcfg_id
    else:
        authConfig = QgsAuthMethodConfig()
        auth_manager().loadAuthenticationConfig(authcfg_id, authConfig, True)
        authConfig.setName(authcfg_name)
        authConfig.setConfig('oauth2config', json.dumps(cfgjson))
        if auth_manager().updateAuthenticationConfig(authConfig):
            return authcfg_id
    return None