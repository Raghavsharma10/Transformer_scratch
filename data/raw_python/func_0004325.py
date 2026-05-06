def remove_connection(self, id_interface, back_or_front):
        """
        Remove a connection between two interfaces

        :param id_interface: One side of relation
        :param back_or_front: This side of relation is back(0) or front(1)

        :return: None

        :raise InterfaceInvalidBackFrontError: Front or Back of interfaces not match to remove connection
        :raise InvalidParameterError: Interface id or back or front indicator is none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        msg_err = u'Parameter %s is invalid. Value: %s.'

        if not is_valid_0_1(back_or_front):
            raise InvalidParameterError(
                msg_err %
                ('back_or_front', back_or_front))

        if not is_valid_int_param(id_interface):
            raise InvalidParameterError(
                msg_err %
                ('id_interface', id_interface))

        url = 'interface/%s/%s/' % (str(id_interface), str(back_or_front))

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)