def check_vip_ip(self, ip, id_evip):
        """
        Get a Ipv4 or Ipv6 for Vip request

        :param ip: IPv4 or Ipv6. 'xxx.xxx.xxx.xxx or xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx'

        :return: Dictionary with the following structure:

        ::

            {'ip': {'ip': < ip - octs for ipv4, blocks for ipv6 - >,
            'id': <id>,
            'network4 or network6'}}.

        :raise IpNaoExisteError: Ipv4 or Ipv6 not found.
        :raise EnvironemntVipNotFoundError: Vip environment not found.
        :raise IPNaoDisponivelError: Ip not available for Vip Environment.
        :raise UserNotAuthorizedError: User dont have permission to perform operation.
        :raise InvalidParameterError: Ip string or vip environment is none or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.

        """

        ip_map = dict()
        ip_map['ip'] = ip
        ip_map['id_evip'] = id_evip

        url = "ip/checkvipip/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)