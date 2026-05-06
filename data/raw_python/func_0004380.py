def buscar_por_id(self, id_ambiente):
        """Obtém um ambiente a partir da chave primária (identificador).

        :param id_ambiente: Identificador do ambiente.

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
            'acl_path': < acl_path >,
            'ipv4_template': < ipv4_template >,
            'ipv6_template': < ipv6_template >,
            'ambiente_rede': < ambiente_rede >}}

        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise InvalidParameterError: Identificador do ambiente é nulo ou inválido.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """
        if not is_valid_int_param(id_ambiente):
            raise InvalidParameterError(
                u'O identificador do ambiente é inválido ou não foi informado.')

        url = 'environment/id/' + str(id_ambiente) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)