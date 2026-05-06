def find(self, ip):
        '''
        Find the abuse contact for a IP address

        :param ip: IPv4 or IPv6 address to check
        :type ip: string

        :returns: emails associated with IP
        :rtype: list
        :returns: none if no contact could be found
        :rtype: None

        :raises: :py:class:`ValueError`: if ip is not properly formatted
        '''
        ip = ipaddr.IPAddress(ip)
        rev = reversename(ip.exploded)
        revip, _ = rev.split(3)
        lookup = revip.concatenate(self.provider).to_text()

        contacts = self._get_txt_record(lookup)
        if contacts:
            return contacts.split(',')