def buscar_por_equipamento(self, nome_equipamento, ip_equipamento):
        """Obtém um ambiente a partir do ip e nome de um equipamento.

        :param nome_equipamento: Nome do equipamento.
        :param ip_equipamento: IP do equipamento no formato XXX.XXX.XXX.XXX.

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

        :raise IpError: IP não cadastrado para o equipamento.
        :raise InvalidParameterError: O nome e/ou o IP do equipamento são vazios ou nulos, ou o IP é inválido.
        :raise EquipamentoNaoExisteError: Equipamento não cadastrado.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if nome_equipamento == '' or nome_equipamento is None:
            raise InvalidParameterError(
                u'O nome do equipamento não foi informado.')

        if not is_valid_ip(ip_equipamento):
            raise InvalidParameterError(
                u'O IP do equipamento é inválido ou não foi informado.')

        url = 'ambiente/equipamento/' + \
            urllib.quote(nome_equipamento) + '/ip/' + str(ip_equipamento) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)