def get_config_tbk(self, confirmation_url):
        '''
        Returns a string with the ``TBK_CONFIG.dat``.

        :param confirmation_url: URL where callback is made.
        '''
        config = (
            "IDCOMERCIO = {commerce_id}\n"
            "MEDCOM = 1\n"
            "TBK_KEY_ID = 101\n"
            "PARAMVERIFCOM = 1\n"
            "URLCGICOM = {confirmation_path}\n"
            "SERVERCOM = {confirmation_host}\n"
            "PORTCOM = {confirmation_port}\n"
            "WHITELISTCOM = ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 0123456789./:=&?_\n"
            "HOST = {confirmation_host}\n"
            "WPORT = {confirmation_port}\n"
            "URLCGITRA = /filtroUnificado/bp_revision.cgi\n"
            "URLCGIMEDTRA = /filtroUnificado/bp_validacion.cgi\n"
            "SERVERTRA = {webpay_server}\n"
            "PORTTRA = {webpay_port}\n"
            "PREFIJO_CONF_TR = HTML_\n"
            "HTML_TR_NORMAL = http://127.0.0.1/notify\n"
        )
        confirmation_uri = six.moves.urllib.parse.urlparse(confirmation_url)
        webpay_server = "https://certificacion.webpay.cl" if self.testing else "https://webpay.transbank.cl"
        webpay_port = 6443 if self.testing else 443
        return config.format(commerce_id=self.id,
                             confirmation_path=confirmation_uri.path,
                             confirmation_host=confirmation_uri.hostname,
                             confirmation_port=confirmation_uri.port,
                             webpay_port=webpay_port,
                             webpay_server=webpay_server)