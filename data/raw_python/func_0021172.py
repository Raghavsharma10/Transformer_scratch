def clean_options(self,
                      using_keytab=False, principal=None,
                      keytab_file=None, ccache_file=None,
                      password=None):
        """Clean argument to related object

        :param bool using_keytab: refer to ``krbContext.__init__``.
        :param str principal: refer to ``krbContext.__init__``.
        :param str keytab_file: refer to ``krbContext.__init__``.
        :param str ccache_file: refer to ``krbContext.__init__``.
        :param str password: refer to ``krbContext.__init__``.

        :return: a mapping containing cleaned names and values, which are used
            internally.
        :rtype: dict
        :raises ValueError: principal is missing or given keytab file does not
            exist, when initialize from a keytab.
        """
        cleaned = {}

        if using_keytab:
            if principal is None:
                raise ValueError('Principal is required when using key table.')
            princ_name = gssapi.names.Name(
                principal, gssapi.names.NameType.kerberos_principal)

            if keytab_file is None:
                cleaned['keytab'] = DEFAULT_KEYTAB
            elif not os.path.exists(keytab_file):
                raise ValueError(
                    'Keytab file {0} does not exist.'.format(keytab_file))
            else:
                cleaned['keytab'] = keytab_file
        else:
            if principal is None:
                principal = get_login()
            princ_name = gssapi.names.Name(principal,
                                           gssapi.names.NameType.user)

        cleaned['using_keytab'] = using_keytab
        cleaned['principal'] = princ_name
        cleaned['ccache'] = ccache_file or DEFAULT_CCACHE
        cleaned['password'] = password

        return cleaned