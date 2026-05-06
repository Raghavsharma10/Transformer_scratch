def set_packet_headers(self, headers):
        """ Set packet header.

        The method will try to set ps_headerprotocol to inform the Xena GUI and tester how to interpret the packet
        header byte sequence specified with PS_PACKETHEADER.
        This is mainly for information purposes, and the stream will transmit the packet header bytes even if no
        protocol segments are specified.
        If the method fails to set some segment it will log a warning and skip setup.

        :param headers: current packet headers
        :type headers: pypacker.layer12.ethernet.Ethernet
        """

        bin_headers = '0x' + binascii.hexlify(headers.bin()).decode('utf-8')
        self.set_attributes(ps_packetheader=bin_headers)

        body_handler = headers
        ps_headerprotocol = []
        while body_handler:
            segment = pypacker_2_xena.get(str(body_handler).split('(')[0].lower(), None)
            if not segment:
                self.logger.warning('pypacker header {} not in conversion list'.format(segment))
                return
            ps_headerprotocol.append(segment)
            if type(body_handler) is Ethernet and body_handler.vlan:
                ps_headerprotocol.append('vlan')
            body_handler = body_handler.body_handler
        self.set_attributes(ps_headerprotocol=' '.join(ps_headerprotocol))