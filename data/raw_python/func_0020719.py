def __search_cert_key(self):
        """
        Get the user credentials if they exist, otherwise throw an exception.
        This code was modified from DBSAPI/dbsHttpService.py and WMCore/Services/Requests.py
        """
        # Now we're trying to guess what the right cert/key combo is...
        # First preference to HOST Certificate, This is how it set in Tier0
        if 'X509_HOST_CERT' in os.environ:
            self._ssl_cert = os.environ['X509_HOST_CERT']
            self._ssl_key = os.environ['X509_HOST_KEY']

        # Second preference to User Proxy, very common
        elif 'X509_USER_PROXY' in os.environ and os.path.exists(os.environ['X509_USER_PROXY']):
            self._ssl_cert = os.environ['X509_USER_PROXY']
            self._ssl_key = self._ssl_cert

        # Third preference to User Cert/Proxy combinition
        elif 'X509_USER_CERT' in os.environ and 'X509_USER_KEY' in os.environ:
            self._ssl_cert = os.environ['X509_USER_CERT']
            self._ssl_key = os.environ['X509_USER_KEY']

        # TODO: only in linux, unix case, add other os case
        # look for proxy at default location /tmp/x509up_u$uid
        elif os.path.exists('/tmp/x509up_u%s' % str(os.getuid())):
            self._ssl_cert = '/tmp/x509up_u%s' % str(os.getuid())
            self._ssl_key = self._ssl_cert

        elif sys.stdin.isatty():
            home_dir = os.environ['HOME']
            user_cert = os.path.join(home_dir, '.globus/usercert.pem')
            user_key = os.path.join(home_dir, '.globus/userkey.pem')

            if os.path.exists(user_cert):
                self._ssl_cert = user_cert
                if os.path.exists(user_key):
                    self._ssl_key = user_key
                    #store password for convenience
                    self._ssl_key_pass = getpass("Password for %s: " % self._ssl_key)
                else:
                    self._ssl_key = self._ssl_cert
            else:
                raise ClientAuthException("No valid X509 cert-key-pair found.")    

        else:
            raise ClientAuthException("No valid X509 cert-key-pair found.")