def set_server_verification_params(
            self, digest_algo=None, dir_cert=None, tolerance=None, no_check_uid=None,
            dir_credentials=None, pass_unix_credentials=None):
        """Sets peer verification params for subscription server.

        These are for secured subscriptions.

        :param str|unicode digest_algo: Digest algorithm. Example: SHA1

            .. note:: Also requires ``dir_cert`` to be set.

        :param str|unicode dir_cert: Certificate directory.

            .. note:: Also requires ``digest_algo`` to be set.

        :param int tolerance: Maximum tolerance (in seconds) of clock skew for secured subscription system.
            Default: 24h.

        :param str|unicode|int|list[str|unicode|int] no_check_uid: Skip signature check for the specified uids
            when using unix sockets credentials.

        :param str|unicode|list[str|unicode] dir_credentials: Directories to search for subscriptions
            key credentials.

        :param bool pass_unix_credentials: Enable management of SCM_CREDENTIALS in subscriptions UNIX sockets.

        """
        if digest_algo and dir_cert:
            self._set('subscriptions-sign-check', '%s:%s' % (digest_algo, dir_cert))

        self._set('subscriptions-sign-check-tolerance', tolerance)
        self._set('subscriptions-sign-skip-uid', no_check_uid, multi=True)
        self._set('subscriptions-credentials-check', dir_credentials, multi=True)
        self._set('subscriptions-use-credentials', pass_unix_credentials, cast=bool)

        return self._section