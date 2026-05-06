def init_kerberos(app, service='HTTP', hostname=gethostname()):
    '''
    Configure the GSSAPI service name, and validate the presence of the
    appropriate principal in the kerberos keytab.

    :param app: a flask application
    :type app: flask.Flask
    :param service: GSSAPI service name
    :type service: str
    :param hostname: hostname the service runs under
    :type hostname: str
    '''
    global _SERVICE_NAME
    _SERVICE_NAME = "%s@%s" % (service, hostname)

    if 'KRB5_KTNAME' not in environ:
        app.logger.warn("Kerberos: set KRB5_KTNAME to your keytab file")
    else:
        try:
            principal = kerberos.getServerPrincipalDetails(service, hostname)
        except kerberos.KrbError as exc:
            app.logger.warn("Kerberos: %s" % exc.message[0])
        else:
            app.logger.info("Kerberos: server is %s" % principal)