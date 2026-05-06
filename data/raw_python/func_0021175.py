def _prepare_context(self):
        """Prepare context

        Initialize credential cache with keytab or password according to
        ``using_keytab`` parameter. Then, ``KRB5CCNAME`` is set properly so
        that Kerberos library called in current context is able to get
        credential from correct cache.

        Internal use only.
        """
        ccache = self._cleaned_options['ccache']

        # Whatever there is KRB5CCNAME was set in current process,
        # original_krb5ccname will contain current value even if None if
        # that variable wasn't set, and when leave krbcontext, it can be
        # handled properly.
        self._original_krb5ccname = os.environ.get(ENV_KRB5CCNAME)

        if ccache == DEFAULT_CCACHE:
            # If user wants to use default ccache, existing KRB5CCNAME in
            # current environment variable should be removed.
            if self._original_krb5ccname:
                del os.environ[ENV_KRB5CCNAME]
        else:
            # When not using default ccache to initialize new credential, let
            # us point to the given ccache by KRB5CCNAME.
            os.environ[ENV_KRB5CCNAME] = ccache

        if self._cleaned_options['using_keytab']:
            self.init_with_keytab()
        else:
            self.init_with_password()