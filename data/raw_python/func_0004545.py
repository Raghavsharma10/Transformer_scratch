def add(self, finalidade_txt, cliente_txt, ambiente_p44_txt, description):
        """Inserts a new Environment VIP and returns its identifier.

        :param finalidade_txt: Finality. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param cliente_txt: ID  Client. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param ambiente_p44_txt: Environment P44. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: Following dictionary:

        ::

            {'environment_vip': {'id': < id >}}

        :raise InvalidParameterError: The value of finalidade_txt, cliente_txt or ambiente_p44_txt is invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        environmentvip_map = dict()
        environmentvip_map['finalidade_txt'] = finalidade_txt
        environmentvip_map['cliente_txt'] = cliente_txt
        environmentvip_map['ambiente_p44_txt'] = ambiente_p44_txt
        environmentvip_map['description'] = description

        code, xml = self.submit(
            {'environment_vip': environmentvip_map}, 'POST', 'environmentvip/')

        return self.response(code, xml)