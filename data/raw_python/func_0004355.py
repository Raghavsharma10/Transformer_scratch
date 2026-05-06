def find_equips(
            self,
            name,
            iexact,
            environment,
            equip_type,
            group,
            ip,
            pagination):
        """
        Find vlans by all search parameters

        :param name: Filter by vlan name column
        :param iexact: Filter by name will be exact?
        :param environment: Filter by environment ID related
        :param equip_type: Filter by equipment_type ID related
        :param group: Filter by equipment group ID related
        :param ip: Filter by each octs in ips related
        :param pagination: Class with all data needed to paginate

        :return: Following dictionary:

        ::

            {'equipamento': {'id': < id_vlan >,
            'nome': < nome_vlan >,
            'num_vlan': < num_vlan >,
            'id_ambiente': < id_ambiente >,
            'descricao': < descricao >,
            'acl_file_name': < acl_file_name >,
            'acl_valida': < acl_valida >,
            'ativada': < ativada >,
            'ambiente_name': < divisao_dc-ambiente_logico-grupo_l3 >
            'redeipv4': [ { all networkipv4 related  } ],
            'redeipv6': [ { all networkipv6 related } ] },
            'total': {< total_registros >} }

        :raise InvalidParameterError: Some parameter was invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not isinstance(pagination, Pagination):
            raise InvalidParameterError(
                u"Invalid parameter: pagination must be a class of type 'Pagination'.")

        equip_map = dict()

        equip_map["start_record"] = pagination.start_record
        equip_map["end_record"] = pagination.end_record
        equip_map["asorting_cols"] = pagination.asorting_cols
        equip_map["searchable_columns"] = pagination.searchable_columns
        equip_map["custom_search"] = pagination.custom_search

        equip_map["nome"] = name
        equip_map["exato"] = iexact
        equip_map["ambiente"] = environment
        equip_map["tipo_equipamento"] = equip_type
        equip_map["grupo"] = group
        equip_map["ip"] = ip

        url = "equipamento/find/"

        code, xml = self.submit({"equipamento": equip_map}, "POST", url)

        key = "equipamento"
        return get_list_map(
            self.response(
                code, xml, [
                    key, "ips", "grupos"]), key)