def as4_capability(self, **kwargs):
        """Set Spanning Tree state.

        Args:
            enabled (bool): Is AS4 Capability enabled? (True, False)
            callback (function): A function executed upon completion of the
                method.  The only parameter passed to `callback` will be the
                ``ElementTree`` `config`.

        Returns:
            Return value of `callback`.

        Raises:

            ValueError: if `enabled` are invalid.

        Examples:
            >>> import pynos.device
            >>> switches = ['10.24.39.211', '10.24.39.203']
            >>> auth = ('admin', 'password')
            >>> for switch in switches:
            ...     conn = (switch, '22')
            ...     with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...         output = dev.bgp.local_asn(local_as='65535',
            ...         rbridge_id='225')
            ...         output = dev.bgp.as4_capability(
            ...         rbridge_id='225', enabled=True)
            ...         output = dev.bgp.as4_capability(
            ...         rbridge_id='225', enabled=False)
        """
        enabled = kwargs.pop('enabled', True)
        callback = kwargs.pop('callback', self._callback)

        if not isinstance(enabled, bool):
            raise ValueError('%s must be `True` or `False`.' % repr(enabled))

        as4_capability_args = dict(vrf_name=kwargs.pop('vrf', 'default'),
                                   rbridge_id=kwargs.pop('rbridge_id', '1'))

        as4_capability = getattr(self._rbridge,
                                 'rbridge_id_router_router_bgp_router_bgp'
                                 '_attributes_capability_as4_enable')

        config = as4_capability(**as4_capability_args)

        if not enabled:
            capability = config.find('.//*capability')
            capability.set('operation', 'delete')
            # shutdown = capability.find('.//*as4-enable')
            # shutdown.set('operation', 'delete')

        return callback(config)