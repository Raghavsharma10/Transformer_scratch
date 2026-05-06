def listar_por_equip(self, equip_id):
        """Lista todos os ambientes por equipamento especifico.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'ambiente': {'id': < id_ambiente >,
            'link': < link >,
            'id_divisao': < id_divisao >,
            'nome_divisao': < nome_divisao >,
            'id_ambiente_logico': < id_ambiente_logico >,
            'nome_ambiente_logico': < nome_ambiente_logico >,
            'id_grupo_l3': < id_grupo_l3 >,
            'nome_grupo_l3': < nome_grupo_l3 >,
            'id_filter': < id_filter >,
            'filter_name': < filter_name >,
            'ambiente_rede': < ambiente_rede >}}

        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if equip_id is None:
            raise InvalidParameterError(
                u'O id do equipamento não foi informado.')

        url = 'ambiente/equip/' + str(equip_id) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)