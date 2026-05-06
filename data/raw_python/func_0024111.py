def make_secure_adaptor(service, mod, client_id, client_secret, tok_update_sec=None):
        """
        :param service: Service to wrap in.
        :param mod: Name (type) of token refresh backend.
        :param client_id: Client identifier.
        :param client_secret: Client secret.
        :param tok_update_sec: Token update interval in seconds.
        """
        if mod == 'TVM':
            return SecureServiceAdaptor(service, TVM(client_id, client_secret), tok_update_sec)

        return SecureServiceAdaptor(service, Promiscuous(), tok_update_sec)