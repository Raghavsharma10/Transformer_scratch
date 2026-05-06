def max_paths(self, **kwargs):
        """Set BGP max paths property.

        Args:
            vrf (str): The VRF for this BGP process.
            rbridge_id (str): The rbridge ID of the device on which BGP will be
                configured in a VCS fabric.
            paths (str): Number of paths for BGP ECMP (default: 8).
            afi (str): Address family to configure. (ipv4, ipv6)
            get (bool): Get config instead of editing config. (True, False)
            callback (function): A function executed upon completion of the
                method.  The only parameter passed to `callback` will be the
                ``ElementTree`` `config`.

        Returns:
            Return value of `callback`.

        Raises:
            ``AttributeError``: When `afi` is not one of ['ipv4', 'ipv6']

        Examples:
            >>> import pynos.device
            >>> conn = ('10.24.39.203', '22')
            >>> auth = ('admin', 'password')
            >>> with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...     output = dev.bgp.max_paths(paths='8',
            ...     rbridge_id='225')
            ...     output = dev.bgp.max_paths(paths='8',
            ...     rbridge_id='225', get=True)
            ...     output = dev.bgp.max_paths(paths='8',
            ...     rbridge_id='225', delete=True)
            ...     output = dev.bgp.max_paths(paths='8', afi='ipv6',
            ...     rbridge_id='225')
            ...     output = dev.bgp.max_paths(paths='8', afi='ipv6',
            ...     rbridge_id='225', get=True)
            ...     output = dev.bgp.max_paths(paths='8', afi='ipv6',
            ...     rbridge_id='225', delete=True)
            ...     output = dev.bgp.max_paths(paths='8', afi='ipv5',
            ...     rbridge_id='225') # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AttributeError
        """
        afi = kwargs.pop('afi', 'ipv4')
        callback = kwargs.pop('callback', self._callback)
        if afi not in ['ipv4', 'ipv6']:
            raise AttributeError('Invalid AFI.')
        args = dict(rbridge_id=kwargs.pop('rbridge_id', '1'),
                    load_sharing_value=kwargs.pop('paths', '8'))
        max_paths = getattr(self._rbridge,
                            'rbridge_id_router_router_bgp_address_family_{0}_'
                            '{0}_unicast_default_vrf_af_common_cmds_holder_'
                            'maximum_paths_load_sharing_value'.format(afi))
        config = max_paths(**args)
        if kwargs.pop('get', False):
            return callback(config, handler='get_config')
        if kwargs.pop('delete', False):
            tag = 'maximum-paths'
            config.find('.//*%s' % tag).set('operation', 'delete')
        return callback(config)