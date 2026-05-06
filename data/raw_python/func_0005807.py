def set_sni_params(self, name, cert, key, ciphers=None, client_ca=None, wildcard=False):
        """Allows setting Server Name Identification (virtual hosting for SSL nodes) params.

        * http://uwsgi.readthedocs.io/en/latest/SNI.html

        :param str|unicode name: Node/server/host name.

        :param str|unicode cert: Certificate file.

        :param str|unicode key: Private key file.

        :param str|unicode ciphers: Ciphers [alias] string.

            Example:
                * DEFAULT
                * HIGH
                * DHE, EDH

            * https://www.openssl.org/docs/man1.1.0/apps/ciphers.html

        :param str|unicode client_ca: Client CA file for client-based auth.

            .. note: You can prepend ! (exclamation mark) to make client certificate
                authentication mandatory.

        :param bool wildcard: Allow regular expressions in ``name`` (used for wildcard certificates).

        """
        command = 'sni'

        if wildcard:
            command += '-regexp'

        args = [item for item in (cert, key, ciphers, client_ca) if item is not None]

        self._set(command, '%s %s' % (name, ','.join(args)))

        return self._section