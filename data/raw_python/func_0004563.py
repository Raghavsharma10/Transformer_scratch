def get_ipv4_or_ipv6(self, ip):
        """
        Get a Ipv4 or Ipv6 by IP

        :param ip: IPv4 or Ipv6. 'xxx.xxx.xxx.xxx or xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx'

        :return: Dictionary with the following structure:

        ::

            {'ips': [{'oct4': < oct4 >, 'oct2': < oct2 >, 'oct3': < oct3 >,
            'oct1': < oct1 >, 'version': < version >,
            'networkipv4': < networkipv4 >, 'id': < id >, 'descricao': < descricao >}, ... ] }.
            or
            {'ips': [ {'block1': < block1 >, 'block2': < block2 >, 'block3': < block3 >, 'block4': < block4 >, 'block5': < block5 >, 'block6': < block6 >, 'block7': < block7 >, 'block8': < block8 >,
            'version': < version >, 'networkipv6': < networkipv6 >, 'id': < id >, 'descricao': < descricao >}, ... ] }.

        :raise IpNaoExisteError: Ipv4 or Ipv6 not found.
        :raise UserNotAuthorizedError: User dont have permission to perform operation.
        :raise InvalidParameterError: Ip string is none or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.
        """

        ip_map = dict()
        ip_map['ip'] = ip

        url = "ip/getbyoctblock/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)