def auth(self, transport, account_name, password):
        """
        Authenticates using username and password.
        """
        auth_token = AuthToken()
        auth_token.account_name = account_name

        attrs = {sconstant.A_BY: sconstant.V_NAME}
        account = SOAPpy.Types.stringType(data=account_name, attrs=attrs)

        params = {sconstant.E_ACCOUNT: account,
                  sconstant.E_PASSWORD: password}

        self.log.debug('Authenticating account %s' % account_name)
        try:
            res = transport.invoke(zconstant.NS_ZIMBRA_ACC_URL,
                                   sconstant.AuthRequest,
                                   params,
                                   auth_token)
        except SoapException as exc:
            raise AuthException(unicode(exc), exc)

        auth_token.token = res.authToken
        
        if hasattr(res, 'sessionId'):
            auth_token.session_id = res.sessionId

        self.log.info('Authenticated account %s, session id %s'
                      % (account_name, auth_token.session_id))

        return auth_token