def find_ip4_by_id(self, id_ip):
        """
        Get an IP by ID

        :param id_ip: IP identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            { ips { id: <id_ip4>,
            oct1: <oct1>,
            oct2: <oct2>,
            oct3: <oct3>,
            oct4: <oct4>,
            equipamento: [ {all equipamentos related} ] ,
            descricao: <descricao>} }

        :raise IpNotAvailableError: Network dont have available IP.
        :raise NetworkIPv4NotFoundError: Network was not found.
        :raise UserNotAuthorizedError: User dont have permission to perform operation.
        :raise InvalidParameterError: Ip identifier is none or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.

        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'Ip identifier is invalid or was not informed.')

        url = 'ip/get/' + str(id_ip) + "/"

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)