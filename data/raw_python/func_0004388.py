def alterar(
            self,
            id_ambiente,
            id_grupo_l3,
            id_ambiente_logico,
            id_divisao,
            link,
            id_filter=None,
            acl_path=None,
            ipv4_template=None,
            ipv6_template=None,
            min_num_vlan_1=None,
            max_num_vlan_1=None,
            min_num_vlan_2=None,
            max_num_vlan_2=None,
            vrf=None):
        """Altera os dados de um ambiente a partir do seu identificador.

        :param id_ambiente: Identificador do ambiente.
        :param id_grupo_l3: Identificador do grupo layer 3.
        :param id_ambiente_logico: Identificador do ambiente lógico.
        :param id_divisao: Identificador da divisão data center.
        :param id_filter: Filter identifier.
        :param link: Link
        :param acl_path: Path where the ACL will be stored
        :param ipv4_template: Template that will be used in Ipv6
        :param ipv6_template: Template that will be used in Ipv4
        :param min_num_vlan_1: Min 1 num vlan valid for this environment
        :param max_num_vlan_1: Max 1 num vlan valid for this environment
        :param min_num_vlan_2: Min 2 num vlan valid for this environment
        :param max_num_vlan_2: Max 2 num vlan valid for this environment

        :return: None

        :raise InvalidParameterError: O identificador do ambiente, o identificador do grupo l3, o identificador do ambiente lógico, e/ou o identificador da divisão de data center são nulos ou inválidos.
        :raise GrupoL3NaoExisteError: Grupo layer 3 não cadastrado.
        :raise AmbienteLogicoNaoExisteError: Ambiente lógico não cadastrado.
        :raise DivisaoDcNaoExisteError: Divisão data center não cadastrada.
        :raise AmbienteDuplicadoError: Ambiente com o mesmo id_grupo_l3, id_ambiente_logico e id_divisao já cadastrado.
        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        if not is_valid_int_param(id_ambiente):
            raise InvalidParameterError(
                u'O identificador do ambiente é inválido ou não foi informado.')

        url = 'ambiente/' + str(id_ambiente) + '/'

        ambiente_map = dict()
        ambiente_map['id_grupo_l3'] = id_grupo_l3
        ambiente_map['id_ambiente_logico'] = id_ambiente_logico
        ambiente_map['id_divisao'] = id_divisao
        ambiente_map['id_filter'] = id_filter
        ambiente_map['link'] = link
        ambiente_map['vrf'] = vrf
        ambiente_map['acl_path'] = acl_path
        ambiente_map['ipv4_template'] = ipv4_template
        ambiente_map['ipv6_template'] = ipv6_template
        ambiente_map['min_num_vlan_1'] = min_num_vlan_1
        ambiente_map['max_num_vlan_1'] = max_num_vlan_1
        ambiente_map['min_num_vlan_2'] = min_num_vlan_2
        ambiente_map['max_num_vlan_2'] = max_num_vlan_2

        code, xml = self.submit({'ambiente': ambiente_map}, 'PUT', url)

        return self.response(code, xml)