def redistribute(self, **kwargs):
        """Set BGP redistribute properties.

        Args:
            vrf (str): The VRF for this BGP process.
            rbridge_id (str): The rbridge ID of the device on which BGP will be
                configured in a VCS fabric.
            source (str): Source for redistributing. (connected)
            afi (str): Address family to configure. (ipv4, ipv6)
            get (bool): Get config instead of editing config. (True, False)
            callback (function): A function executed upon completion of the
                method.  The only parameter passed to `callback` will be the
                ``ElementTree`` `config`.

        Returns:
            Return value of `callback`.

        Raises:
            KeyError: if `source` is not specified.

        Examples:
            >>> import pynos.device
            >>> conn = ('10.24.39.203', '22')
            >>> auth = ('admin', 'password')
            >>> with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...     output = dev.bgp.redistribute(source='connected',
            ...     rbridge_id='225')
            ...     output = dev.bgp.redistribute(source='connected',
            ...     rbridge_id='225', get=True)
            ...     output = dev.bgp.redistribute(source='connected',
            ...     rbridge_id='225', delete=True)
            ...     dev.bgp.redistribute() # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            KeyError
            ...     dev.bgp.redistribute(source='connected', rbridge_id='225',
            ...     afi='hodor') # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AttributeError
            ...     dev.bgp.redistribute(source='hodor', rbridge_id='225',
            ...     afi='ipv4') # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AttributeError
        """
        # This method is the same as the base method except for one place.
        # The class doesn't inherit from the base class, though, so we have
        # to duplicate.
        source = kwargs.pop('source')
        afi = kwargs.pop('afi', 'ipv4')
        callback = kwargs.pop('callback', self._callback)
        if afi not in ['ipv4', 'ipv6']:
            raise AttributeError('Invalid AFI.')
        args = dict(rbridge_id=kwargs.pop('rbridge_id', '1'),
                    afi=afi, source=source)
        redistribute = self._redistribute_builder(afi=afi, source=source)
        config = redistribute(**args)
        if kwargs.pop('get', False):
            return callback(config, handler='get_config')
        if kwargs.pop('delete', False):
            tag = 'redistribute-%s' % source
            config.find('.//*%s' % tag).set('operation', 'delete')
        return callback(config)