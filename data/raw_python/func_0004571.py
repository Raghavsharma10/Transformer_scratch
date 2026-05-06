def find_ips_by_equip(self, id_equip):
        """
        Get Ips related to equipment by its identifier

        :param id_equip: Equipment identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

            { ips: { ipv4:[
            id: <id_ip4>,
            oct1: <oct1>,
            oct2: <oct2>,
            oct3: <oct3>,
            oct4: <oct4>,
            descricao: <descricao> ]
            ipv6:[
            id: <id_ip6>,
            block1: <block1>,
            block2: <block2>,
            block3: <block3>,
            block4: <block4>,
            block5: <block5>,
            block6: <block6>,
            block7: <block7>,
            block8: <block8>,
            descricao: <descricao> ] } }

        :raise UserNotAuthorizedError: User dont have permission to list ips.
        :raise InvalidParameterError: Equipment identifier is none or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.

        """
        url = 'ip/getbyequip/' + str(id_equip) + "/"

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml, ['ipv4', 'ipv6'])