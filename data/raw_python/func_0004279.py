def add_network(self, network, id_vlan, id_network_type,
                    id_environment_vip=None, cluster_unit=None):
        """
        Add new network

        :param network: IPV4 or IPV6 (ex.: "10.0.0.3/24")
        :param id_vlan: Identifier of the Vlan. Integer value and greater than zero.
        :param id_network_type: Identifier of the NetworkType. Integer value and greater than zero.
        :param id_environment_vip: Identifier of the Environment Vip. Integer value and greater than zero.

        :return: Following dictionary:

        ::

          {'network':{'id': <id_network>,
          'rede': <network>,
          'broadcast': <broadcast> (if is IPv4),
          'mask': <net_mask>,
          'id_vlan': <id_vlan>,
          'id_tipo_rede': <id_network_type>,
          'id_ambiente_vip': <id_ambiente_vip>,
          'active': <active>} }

        :raise TipoRedeNaoExisteError: NetworkType not found.
        :raise InvalidParameterError: Invalid ID for Vlan or NetworkType.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise IPNaoDisponivelError: Network address unavailable to create a NetworkIPv4.
        :raise NetworkIPRangeEnvError: Other environment already have this ip range.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        network_map = dict()
        network_map['network'] = network
        network_map['id_vlan'] = id_vlan
        network_map['id_network_type'] = id_network_type
        network_map['id_environment_vip'] = id_environment_vip
        network_map['cluster_unit'] = cluster_unit

        code, xml = self.submit(
            {'network': network_map}, 'POST', 'network/add/')

        return self.response(code, xml)