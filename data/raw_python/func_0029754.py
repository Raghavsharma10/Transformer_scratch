def _multihop_xml(self, **kwargs):
        """Build BGP multihop XML.

        Do not use this method directly.  You probably want ``multihop``.

        Args:
            rbridge_id (str): The rbridge ID of the device on which BGP will be
                configured in a VCS fabric.
            neighbor (ipaddress.ip_interface): `ip_interface` object containing
                peer IP address (IPv4 or IPv6).
            count (str): Number of hops to allow. (1-255)

        Returns:
            ``ElementTree``: XML for configuring BGP multihop.

        Raises:
            KeyError: if any arg is not specified.

        Examples:
            >>> import pynos.device
            >>> from ipaddress import ip_interface
            >>> conn = ('10.24.39.230', '22')
            >>> auth = ('admin', 'password')
            >>> with pynos.device.Device(conn=conn, auth=auth) as dev:
            ...     dev.bgp._multihop_xml(neighbor=ip_interface(unicode(
            ...     '10.10.10.10')), count='5', vrf='default', rbridge_id='1')
            ...     dev.bgp._multihop_xml(
            ...     ip='10.10.10.10') # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            KeyError
        """
        ip_addr = kwargs.pop('neighbor')
        ip = str(ip_addr.ip)
        rbr_ns = 'urn:brocade.com:mgmt:brocade-rbridge'
        bgp_ns = 'urn:brocade.com:mgmt:brocade-bgp'
        config = ET.Element('config')
        ele = ET.SubElement(config, 'rbridge-id', xmlns=rbr_ns)
        ET.SubElement(ele, 'rbridge-id').text = kwargs.pop('rbridge_id')
        ele = ET.SubElement(ele, 'router')
        ele = ET.SubElement(ele, 'router-bgp', xmlns=bgp_ns)
        ele = ET.SubElement(ele, 'router-bgp-attributes')
        ele = ET.SubElement(ele, 'neighbor')
        if ip_addr.version == 4:
            ele = ET.SubElement(ele, 'neighbor-ips')
            ele = ET.SubElement(ele, 'neighbor-addr')
            ET.SubElement(ele, 'router-bgp-neighbor-address').text = ip
        else:
            ele = ET.SubElement(ele, 'neighbor-ipv6s')
            ele = ET.SubElement(ele, 'neighbor-ipv6-addr')
            ET.SubElement(ele, 'router-bgp-neighbor-ipv6-address').text = ip
        ele = ET.SubElement(ele, 'ebgp-multihop')
        ET.SubElement(ele, 'ebgp-multihop-count').text = kwargs.pop('count')
        return config