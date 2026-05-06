def get_ip_by_equip_and_vip(self, equip_name, id_evip):
        """
        Get a available IP in the Equipment related Environment VIP

        :param equip_name: Equipment Name.
        :param id_evip: Vip environment identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            { 'ipv4': [ {'id': < id >, 'ip': < ip >, 'network': { 'id': < id >, 'network': < network >, 'mask': < mask >, }} ... ],
            'ipv6': [ {'id': < id >, 'ip': < ip >, 'network': { 'id': < id >, 'network': < network >, 'mask': < mask >, }} ... ] }

        :raise InvalidParameterError: Vip environment identifier or equipment name is none or invalid.
        :raise EquipamentoNotFoundError: Equipment not registered.
        :raise EnvironmentVipNotFoundError: Vip environment not registered.
        :raise UserNotAuthorizedError: User dont have permission to perform operation.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.
        """

        if not is_valid_int_param(id_evip):
            raise InvalidParameterError(
                u'Vip environment is invalid or was not informed.')

        ip_map = dict()
        ip_map['equip_name'] = equip_name
        ip_map['id_evip'] = id_evip

        url = "ip/getbyequipandevip/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)