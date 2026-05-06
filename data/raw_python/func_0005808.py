def set_sni_dir_params(self, dir, ciphers=None):
        """Enable checking for cert/key/client_ca file in the specified directory
        and create a sni/ssl context on demand.

        Expected filenames:
            * <sni-name>.crt
            * <sni-name>.key
            * <sni-name>.ca - this file is optional

        * http://uwsgi.readthedocs.io/en/latest/SNI.html#massive-sni-hosting

        :param str|unicode dir:

        :param str|unicode ciphers: Ciphers [alias] string.

            Example:
                * DEFAULT
                * HIGH
                * DHE, EDH

            * https://www.openssl.org/docs/man1.1.0/apps/ciphers.html

        """
        self._set('sni-dir', dir)
        self._set('sni-dir-ciphers', ciphers)

        return self._section