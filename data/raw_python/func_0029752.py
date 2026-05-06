def _redistribute_builder(self, afi='ipv4', source=None):
        """Build BGP redistribute method.

        Do not use this method directly.  You probably want ``redistribute``.

        Args:
            source (str): Source for redistributing. (connected)
            afi (str): Address family to configure. (ipv4, ipv6)

        Returns:
            Method to redistribute desired source.

        Raises:
            KeyError: if `source` is not specified.

        Examples:
            >>> import pynos.device
            >>> conn = ('10.24.39.203', '22')
            >>> auth = ('admin', 'password')
            >>> with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...     output = dev.bgp._redistribute_builder(source='connected',
            ...     afi='ipv4')
            ...     dev.bgp._redistribute_builder(source='hodor',
            ...     afi='ipv4') # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AttributeError
        """
        if source == 'connected':
            return getattr(self._rbridge,
                           'rbridge_id_router_router_bgp_address_family_{0}_'
                           '{0}_unicast_default_vrf_af_{0}_uc_and_vrf_cmds_'
                           'call_point_holder_redistribute_connected_'
                           'redistribute_connected'.format(afi))
        # TODO: Add support for 'static' and 'ospf'
        else:
            raise AttributeError('Invalid source.')