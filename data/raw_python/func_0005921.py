def subscribe(
            self, server=None, key=None, address=None, address_vassal=None,
            balancing_weight=None, balancing_algo=None, modifier=None, signing=None, check_file=None, protocol=None,
            sni_cert=None, sni_key=None, sni_client_ca=None):
        """Registers a subscription intent.

        :param str|unicode server: Subscription server address (UDP or UNIX socket).

            Examples:
                * 127.0.0.1:7171

        :param str|unicode key: Key to subscribe. Generally the domain name (+ optional '/< mountpoint>').
            Examples:
                * mydomain.it/foo
                * mydomain.it/foo/bar (requires ``mountpoints_depth=2``)
                * mydomain.it
                * ubuntu64.local:9090

        :param str|unicode|int address: Address to subscribe (the value for the key)
            or zero-based internal socket number (integer).

        :param str|unicode address: Vassal node address.

        :param int balancing_weight: Load balancing value. Default: 1.

        :param balancing_algo: Load balancing algorithm to use. See ``balancing_algorithms``
            .. note:: Since 2.1

        :param Modifier modifier: Routing modifier object. See ``.routing.modifiers``

        :param list|tuple signing:  Signing basics, expects two elements list/tuple:
            (signing_algorithm, key).

            Examples:
                * SHA1:idlessh001

        :param str|unicode check_file: If this file exists the subscription packet is sent,
            otherwise it is skipped.

        :param str|unicode protocol: the protocol to use, by default it is ``uwsgi``.
            See ``.networking.socket_types``.

            .. note:: Since 2.1

        :param str|unicode sni_cert: Certificate file to use for SNI proxy management.
            * http://uwsgi.readthedocs.io/en/latest/SNI.html#subscription-system-and-sni

        :param str|unicode sni_key: sni_key Key file to use for SNI proxy management.
            * http://uwsgi.readthedocs.io/en/latest/SNI.html#subscription-system-and-sni

        :param str|unicode sni_client_ca: Ca file to use for SNI proxy management.
            * http://uwsgi.readthedocs.io/en/latest/SNI.html#subscription-system-and-sni

        """
        # todo params: inactive (inactive slot activation)

        if not any((server, key)):
            raise ConfigurationError('Subscription requires `server` or `key` to be set.')

        address_key = 'addr'
        if isinstance(address, int):
            address_key = 'socket'

        if balancing_algo:
            backup = getattr(balancing_algo, 'backup_level', None)

        if signing:
            signing = ':'.join(signing)

        if modifier:
            modifier1 = modifier
            if modifier.submod:
                modifier2 = modifier.submod

        rule = KeyValue(
            filter_locals(locals(), drop=['address_key', 'modifier']),
            aliases={
                'address': address_key,
                'address_vassal': 'vassal',
                'signing': 'sign',
                'check_file': 'check',
                'balancing_weight': 'weight',
                'balancing_algo': 'algo',
                'protocol': 'proto',
                'sni_cert': 'sni_crt',
                'sni_client_ca': 'sni_ca',
            },
        )

        self._set('subscribe2', rule)

        return self._section