def from_url(cls, reactor, url, key, alg=RS256, jws_client=None):
        """
        Construct a client from an ACME directory at a given URL.

        :param url: The ``twisted.python.url.URL`` to fetch the directory from.
            See `txacme.urls` for constants for various well-known public
            directories.
        :param reactor: The Twisted reactor to use.
        :param ~josepy.jwk.JWK key: The client key to use.
        :param alg: The signing algorithm to use.  Needs to be compatible with
            the type of key used.
        :param JWSClient jws_client: The underlying client to use, or ``None``
            to construct one.

        :return: The constructed client.
        :rtype: Deferred[`Client`]
        """
        action = LOG_ACME_CONSUME_DIRECTORY(
            url=url, key_type=key.typ, alg=alg.name)
        with action.context():
            check_directory_url_type(url)
            jws_client = _default_client(jws_client, reactor, key, alg)
            return (
                DeferredContext(jws_client.get(url.asText()))
                .addCallback(json_content)
                .addCallback(messages.Directory.from_json)
                .addCallback(
                    tap(lambda d: action.add_success_fields(directory=d)))
                .addCallback(cls, reactor, key, jws_client)
                .addActionFinish())