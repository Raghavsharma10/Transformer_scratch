def create_domain(self, name, ip_address, **kwargs):
        """
        Add a new domain name resource to the account.

        Note that this method does not actually register a new domain name; it
        merely configures DigitalOcean's nameservers to provide DNS resolution
        for the domain.  See `How To Set Up a Host Name with DigitalOcean
        <https://www.digitalocean.com/community/tutorials/how-to-set-up-a-host-name-with-digitalocean>`_
        for more information.

        :param str name: the domain name to add
        :param ip_address: the IP address to which the domain should point
        :type ip_address: string or `FloatingIP`
        :param kwargs: additional fields to include in the API request
        :return: the new domain resource
        :rtype: Domain
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if isinstance(ip_address, FloatingIP):
            ip_address = ip_address.ip
        data = {"name": name, "ip_address": ip_address}
        data.update(kwargs)
        return self._domain(self.request('/v2/domains', method='POST',
                                                        data=data)["domain"])