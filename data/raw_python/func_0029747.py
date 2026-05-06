def local_asn(self, **kwargs):
        """Set BGP local ASN.

        Args:
            local_as (str): Local ASN of NOS deice.
            rbridge_id (str): The rbridge ID of the device on which BGP will be
                configured in a VCS fabric.
            get (bool): Get config instead of editing config. (True, False)
            callback (function): A function executed upon completion of the
                method.  The only parameter passed to `callback` will be the
                ``ElementTree`` `config`.

        Returns:
            Return value of `callback`.

        Raises:
            KeyError: if `local_as` is not specified.

        Examples:
            >>> import pynos.device
            >>> conn = ('10.24.39.203', '22')
            >>> auth = ('admin', 'password')
            >>> with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...     output = dev.bgp.local_asn(local_as='65535',
            ...     rbridge_id='225')
            ...     dev.bgp.local_asn() # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            KeyError
        """
        is_get_config = kwargs.pop('get', False)
        if not is_get_config:
            local_as = kwargs.pop('local_as')
        else:
            local_as = ''
        rbridge_id = kwargs.pop('rbridge_id', '1')
        callback = kwargs.pop('callback', self._callback)
        local_as_args = dict(local_as=local_as,
                             rbridge_id=rbridge_id)
        enable_bgp = getattr(self._rbridge,
                             'rbridge_id_router_router_bgp_router_bgp_'
                             'attributes_local_as')(**local_as_args)
        bgp = enable_bgp.find('.//*.//*.//*')
        bgp.remove(bgp.find('.//*'))
        if not is_get_config:
            callback(enable_bgp)
        local_as = getattr(self._rbridge,
                           'rbridge_id_router_router_bgp_router_bgp_attri'
                           'butes_local_as')
        config = local_as(**local_as_args)
        if is_get_config:
            return callback(config, handler='get_config')
        return callback(config)