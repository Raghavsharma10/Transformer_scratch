def search(
            self,
            id_environment_vip=None,
            finalidade_txt=None,
            cliente_txt=None,
            ambiente_p44_txt=None):
        """Search Environment VIP from by parameters.

        Case the id parameter has been passed, the same it has priority over the other parameters.

        :param id_environment_vip: Identifier of the Environment VIP. Integer value and greater than zero.
        :param finalidade_txt: Finality. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param cliente_txt: ID  Client. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param ambiente_p44_txt: Environment P44. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: Following dictionary:

        ::

            {‘environment_vip’:
            {‘id’: < id >,
            ‘finalidade_txt’: < finalidade_txt >,
            ‘finalidade’: < finalidade >,
            ‘cliente_txt’: < cliente_txt >,
            ‘ambiente_p44_txt’: < ambiente_p44_txt >}}

        :raise InvalidParameterError: The value of id_environment_vip, finalidade_txt, cliente_txt or ambiente_p44_txt is invalid.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        environmentvip_map = dict()
        environmentvip_map['id_environment_vip'] = id_environment_vip
        environmentvip_map['finalidade_txt'] = finalidade_txt
        environmentvip_map['cliente_txt'] = cliente_txt
        environmentvip_map['ambiente_p44_txt'] = ambiente_p44_txt

        code, xml = self.submit(
            {'environment_vip': environmentvip_map}, 'POST', 'environmentvip/search/')

        return self.response(code, xml)