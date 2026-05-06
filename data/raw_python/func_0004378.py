def listar(self, id_divisao=None, id_ambiente_logico=None):
        """Lista os ambientes filtrados conforme parâmetros informados.

        Se os dois parâmetros têm o valor None então retorna todos os ambientes.
        Se o id_divisao é diferente de None então retorna os ambientes filtrados
        pelo valor de id_divisao.
        Se o id_divisao e id_ambiente_logico são diferentes de None então retorna
        os ambientes filtrados por id_divisao e id_ambiente_logico.

        :param id_divisao: Identificador da divisão de data center.
        :param id_ambiente_logico: Identificador do ambiente lógico.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'ambiente': [{'id': < id_ambiente >,
            'link': < link >,
            'id_divisao': < id_divisao >,
            'nome_divisao': < nome_divisao >,
            'id_ambiente_logico': < id_ambiente_logico >,
            'nome_ambiente_logico': < nome_ambiente_logico >,
            'id_grupo_l3': < id_grupo_l3 >,
            'nome_grupo_l3': < nome_grupo_l3 >,
            'id_filter': < id_filter >,
            'filter_name': < filter_name >,
            'ambiente_rede': < ambiente_rede >},
            ... demais ambientes ... ]}

        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        url = 'ambiente/'

        if is_valid_int_param(id_divisao) and not is_valid_int_param(
                id_ambiente_logico):
            url = 'ambiente/divisao_dc/' + str(id_divisao) + '/'
        elif is_valid_int_param(id_divisao) and is_valid_int_param(id_ambiente_logico):
            url = 'ambiente/divisao_dc/' + \
                str(id_divisao) + '/ambiente_logico/' + str(id_ambiente_logico) + '/'

        code, xml = self.submit(None, 'GET', url)

        key = 'ambiente'
        return get_list_map(self.response(code, xml, [key]), key)