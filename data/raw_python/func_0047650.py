def pre_auth(self, transport, account_name):
        """
        Authenticates using username and domain key.
        """
        auth_token = AuthToken()
        auth_token.account_name = account_name

        domain = util.get_domain(account_name)
        if domain == None:
            raise AuthException('Invalid auth token account')

        if domain in self.domains:
            domain_key = self.domains[domain]
        else:
            domain_key = None

        if domain_key == None:
            raise AuthException('Invalid domain key for domain %s' % domain)

        self.log.debug('Initialized domain key for account %s'
                      % account_name)

        expires = 0
        timestamp = int(time() * 1000)
        pak = hmac.new(domain_key, '%s|%s|%s|%s' %
                       (account_name, sconstant.E_NAME, expires, timestamp),
                       hashlib.sha1).hexdigest()

        attrs = {sconstant.A_BY: sconstant.V_NAME}
        account = SOAPpy.Types.stringType(data=account_name, attrs=attrs)

        attrs = {sconstant.A_TIMESTAMP: timestamp, sconstant.A_EXPIRES: expires}
        preauth = SOAPpy.Types.stringType(data=pak,
                                          name=sconstant.E_PREAUTH,
                                          attrs=attrs)

        params = {sconstant.E_ACCOUNT: account,
                  sconstant.E_PREAUTH: preauth}

        self.log.debug('Authenticating account %s using domain key'
                       % account_name)
        try:
            res = transport.invoke(zconstant.NS_ZIMBRA_ACC_URL,
                                   sconstant.AuthRequest,
                                   params,
                                   auth_token)
        except SoapException as exc:
            raise AuthException(unicode(exc), exc)

        auth_token.token = res.authToken
        auth_token.session_id = res.sessionId

        self.log.info('Authenticated account %s, session id %s'
                      % (account_name, auth_token.session_id))

        return auth_token