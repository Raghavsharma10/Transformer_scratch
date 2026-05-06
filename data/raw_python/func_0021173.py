def init_with_keytab(self):
        """Initialize credential cache with keytab"""
        creds_opts = {
            'usage': 'initiate',
            'name': self._cleaned_options['principal'],
        }

        store = {}
        if self._cleaned_options['keytab'] != DEFAULT_KEYTAB:
            store['client_keytab'] = self._cleaned_options['keytab']
        if self._cleaned_options['ccache'] != DEFAULT_CCACHE:
            store['ccache'] = self._cleaned_options['ccache']
        if store:
            creds_opts['store'] = store

        creds = gssapi.creds.Credentials(**creds_opts)
        try:
            creds.lifetime
        except gssapi.exceptions.ExpiredCredentialsError:
            new_creds_opts = copy.deepcopy(creds_opts)
            # Get new credential and put it into a temporary ccache
            if 'store' in new_creds_opts:
                new_creds_opts['store']['ccache'] = _get_temp_ccache()
            else:
                new_creds_opts['store'] = {'ccache': _get_temp_ccache()}
            creds = gssapi.creds.Credentials(**new_creds_opts)
            # Then, store new credential back to original specified ccache,
            # whatever a given ccache file or the default one.
            _store = None
            # If default cccache is used, no need to specify ccache in store
            # parameter passed to ``creds.store``.
            if self._cleaned_options['ccache'] != DEFAULT_CCACHE:
                _store = {'ccache': store['ccache']}
            creds.store(usage='initiate', store=_store, overwrite=True)