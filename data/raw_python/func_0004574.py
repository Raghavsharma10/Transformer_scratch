def delete_ip4(self, id_ip):
        """
        Delete an IP4

        :param id_ip: Ipv4 identifier. Integer value and greater than zero.

        :return: None

        :raise IpNotFoundError: IP is not registered.
        :raise DataBaseError: Networkapi failed to access the database.

        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(u'Ipv4 identifier is invalid or was not informed.')

        url = 'ip4/delete/' + str(id_ip) + "/"
        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)