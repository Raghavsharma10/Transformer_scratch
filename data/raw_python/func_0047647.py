def authenticate_admin(self, transport, account_name, password):
        """
        Authenticates administrator using username and password.
        """
        Authenticator.authenticate_admin(self, transport, account_name, password)

        auth_token = AuthToken()
        auth_token.account_name = account_name

        params = {sconstant.E_NAME: account_name,
                  sconstant.E_PASSWORD: password}

        self.log.debug('Authenticating admin %s' % account_name)
        try:
            res = transport.invoke(zconstant.NS_ZIMBRA_ADMIN_URL,
                                   sconstant.AuthRequest,
                                   params,
                                   auth_token)
        except SoapException as exc:
            raise AuthException(unicode(exc), exc)

        auth_token.token = res.authToken
        auth_token.session_id = res.sessionId

        self.log.info('Authenticated admin %s, session id %s'
                      % (account_name, auth_token.session_id))

        return auth_token