def __search_ca_path(self):
        """
        Get CA Path to check the validity of the server host certificate on the client side
        """
        if "X509_CERT_DIR" in os.environ:
            self._ca_path = os.environ['X509_CERT_DIR']

        elif os.path.exists('/etc/grid-security/certificates'):
            self._ca_path = '/etc/grid-security/certificates'

        else:
            raise ClientAuthException("Could not find a valid CA path")