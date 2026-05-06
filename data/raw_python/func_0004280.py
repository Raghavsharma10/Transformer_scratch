def edit_network(self, id_network, ip_type, id_net_type, id_env_vip=None, cluster_unit=None):
        """
        Edit a network 4 or 6

        :param id_network: Identifier of the Network. Integer value and greater than zero.
        :param id_net_type: Identifier of the NetworkType. Integer value and greater than zero.
        :param id_env_vip: Identifier of the Environment Vip. Integer value and greater than zero.
        :param ip_type: Identifier of the Network IP Type: 0 = IP4 Network; 1 = IP6 Network;

        :return: None

        :raise TipoRedeNaoExisteError: NetworkType not found.
        :raise InvalidParameterError: Invalid ID for Vlan or NetworkType.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise IPNaoDisponivelError: Network address unavailable to create a NetworkIPv4.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        net_map = dict()
        net_map['id_network'] = id_network
        net_map['ip_type'] = ip_type
        net_map['id_net_type'] = id_net_type
        net_map['id_env_vip'] = id_env_vip
        net_map['cluster_unit'] = cluster_unit

        code, xml = self.submit({'net': net_map}, 'POST', 'network/edit/')

        return self.response(code, xml)