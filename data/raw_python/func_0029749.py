def remove_bgp(self, **kwargs):
        """Remove BGP process completely.

        Args:
            vrf (str): The VRF for this BGP process.
            rbridge_id (str): The rbridge ID of the device on which BGP will be
                configured in a VCS fabric.
            callback (function): A function executed upon completion of the
                method.  The only parameter passed to `callback` will be the
                ``ElementTree`` `config`.

        Returns:
            Return value of `callback`.

        Raises:
            None

        Examples:
            >>> import pynos.device
            >>> conn = ('10.24.39.203', '22')
            >>> auth = ('admin', 'password')
            >>> with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...     output = dev.bgp.local_asn(local_as='65535',
            ...     rbridge_id='225')
            ...     output = dev.bgp.remove_bgp(rbridge_id='225')
        """
        rbridge_id = kwargs.pop('rbridge_id', '1')
        callback = kwargs.pop('callback', self._callback)
        disable_args = dict(rbridge_id=rbridge_id, local_as='65000')
        config = getattr(self._rbridge,
                         'rbridge_id_router_router_bgp_router_bgp_'
                         'attributes_local_as')(**disable_args)
        bgp = config.find('.//*.//*.//*')
        bgp.remove(bgp.find('.//*'))
        bgp.set('operation', 'delete')

        return callback(config)