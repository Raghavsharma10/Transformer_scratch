def neighbor(self, **kwargs):
        """Add BGP neighbor.

        Args:
            ip_addr (str): IP Address of BGP neighbor.
            remote_as (str): Remote ASN of BGP neighbor.
            rbridge_id (str): The rbridge ID of the device on which BGP will be
                configured in a VCS fabric.
            delete (bool): Deletes the neighbor if `delete` is ``True``.
            get (bool): Get config instead of editing config. (True, False)
            callback (function): A function executed upon completion of the
                method.  The only parameter passed to `callback` will be the
                ``ElementTree`` `config`.

        # Returns:
            Return value of `callback`.

        Raises:
            KeyError: if `remote_as` or `ip_addr` is not specified.

        Examples:
            >>> import pynos.device
            >>> conn = ('10.24.39.203', '22')
            >>> auth = ('admin', 'password')
            >>> with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...     output = dev.bgp.local_asn(local_as='65535',
            ...     rbridge_id='225')
            ...     output = dev.bgp.neighbor(ip_addr='10.10.10.10',
            ...     remote_as='65535', rbridge_id='225')
            ...     output = dev.bgp.neighbor(remote_as='65535',
            ...     rbridge_id='225',
            ...     ip_addr='2001:4818:f000:1ab:cafe:beef:1000:1')
            ...     output = dev.bgp.neighbor(ip_addr='10.10.10.10',
            ...     delete=True, rbridge_id='225')
            ...     output = dev.bgp.neighbor(remote_as='65535',
            ...     rbridge_id='225',
            ...     ip_addr='2001:4818:f000:1ab:cafe:beef:1000:1',
            ...     delete=True)
            ...     dev.bgp.neighbor() # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            KeyError
        """
        ip_addr = kwargs.pop('ip_addr')
        remote_as = kwargs.pop('remote_as', None)
        rbridge_id = kwargs.pop('rbridge_id', '1')
        delete = kwargs.pop('delete', False)
        callback = kwargs.pop('callback', self._callback)
        ip_addr = ip_interface(unicode(ip_addr))

        if not delete and remote_as is None:
            raise ValueError('When configuring a neighbor, you must specify '
                             'its remote-as.')

        neighbor_args = dict(router_bgp_neighbor_address=str(ip_addr.ip),
                             remote_as=remote_as,
                             rbridge_id=rbridge_id)
        if ip_addr.version == 6:
            neighbor_args['router_bgp_neighbor_ipv6_address'] = str(ip_addr.ip)

        if ip_addr.version == 4:
            neighbor = getattr(self._rbridge,
                               'rbridge_id_router_router_bgp_'
                               'router_bgp_attributes_neighbor_neighbor_ips_'
                               'neighbor_addr_remote_as')
            ip_addr_path = './/*remote-as'
        else:
            neighbor = getattr(self._rbridge,
                               'rbridge_id_router_router_bgp_'
                               'router_bgp_attributes_neighbor_'
                               'neighbor_ipv6s_neighbor_ipv6_addr_remote_as')
            ip_addr_path = './/*remote-as'

        config = neighbor(**neighbor_args)

        if delete and config.find(ip_addr_path) is not None:
            if ip_addr.version == 4:
                config.find(ip_addr_path).set('operation', 'delete')
                config.find('.//*router-bgp-neighbor-address').set('operation',
                                                                   'delete')
            elif ip_addr.version == 6:
                config.find(ip_addr_path).set('operation', 'delete')
                config.find('.//*router-bgp-neighbor-ipv6-address').set(
                    'operation', 'delete')
        else:
            if ip_addr.version == 6:
                callback(config)
                activate_args = dict(rbridge_id=rbridge_id,
                                     af_ipv6_neighbor_address=str(ip_addr.ip))
                activate_neighbor = getattr(self._rbridge,
                                            'rbridge_id_router_router_bgp_'
                                            'address_family_ipv6_ipv6_unicast_'
                                            'default_vrf_neighbor_af_ipv6_'
                                            'neighbor_address_holder_af_ipv6_'
                                            'neighbor_address_activate')
                config = activate_neighbor(**activate_args)
        if kwargs.pop('get', False):
            return callback(config, handler='get_config')
        return callback(config)