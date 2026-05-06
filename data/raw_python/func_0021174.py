def init_with_password(self):
        """Initialize credential cache with password

        **Causion:** once you enter password from command line, or pass it to
        API directly, the given password is not encrypted always. Although
        getting credential with password works, from security point of view, it
        is strongly recommended **NOT** use it in any formal production
        environment. If you need to initialize credential in an application to
        application Kerberos authentication context, keytab has to be used.

        :raises IOError: when trying to prompt to input password from command
            line but no attry is available.
        """
        creds_opts = {
            'usage': 'initiate',
            'name': self._cleaned_options['principal'],
        }
        if self._cleaned_options['ccache'] != DEFAULT_CCACHE:
            creds_opts['store'] = {'ccache': self._cleaned_options['ccache']}

        cred = gssapi.creds.Credentials(**creds_opts)
        try:
            cred.lifetime
        except gssapi.exceptions.ExpiredCredentialsError:
            password = self._cleaned_options['password']

            if not password:
                if not sys.stdin.isatty():
                    raise IOError(
                        'krbContext is not running from a terminal. So, you '
                        'need to run kinit with your principal manually before'
                        ' anything goes.')

                # If there is no password specified via API call, prompt to
                # enter one in order to continue to get credential. BUT, in
                # some cases, blocking program and waiting for input of
                # password is really bad, which may be only suitable for some
                # simple use cases, for example, writing some scripts to test
                # something that need Kerberos authentication. Anyway, whether
                # it is really to enter a password from command line, it
                # depends on concrete use cases totally.
                password = getpass.getpass()

            cred = gssapi.raw.acquire_cred_with_password(
                self._cleaned_options['principal'], password)

            ccache = self._cleaned_options['ccache']
            if ccache == DEFAULT_CCACHE:
                gssapi.raw.store_cred(cred.creds,
                                      usage='initiate',
                                      overwrite=True)
            else:
                gssapi.raw.store_cred_into({'ccache': ccache},
                                           cred.creds,
                                           usage='initiate',
                                           overwrite=True)