def configuration_save(
            self,
            id_environment,
            network,
            prefix,
            ip_version,
            network_type):
        """
        Add new prefix configuration

        :param id_environment: Identifier of the Environment. Integer value and greater than zero.
        :param network: Network Ipv4 or Ipv6.
        :param prefix: Prefix 0-32 to Ipv4 or 0-128 to Ipv6.
        :param ip_version: v4 to IPv4 or v6 to IPv6
        :param network_type: type network

        :return: Following dictionary:

        ::

            {'network':{'id_environment': <id_environment>,
            'id_vlan': <id_vlan>,
            'network_type': <network_type>,
            'network': <network>,
            'prefix': <prefix>} }

        :raise ConfigEnvironmentInvalidError: Invalid Environment Configuration or not registered.
        :raise InvalidValueError: Invalid Id for environment or network or network_type or prefix.
        :raise AmbienteNotFoundError: Environment not registered.
        :raise DataBaseError: Failed into networkapi access data base.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        network_map = dict()
        network_map['id_environment'] = id_environment
        network_map['network'] = network
        network_map['prefix'] = prefix
        network_map['ip_version'] = ip_version
        network_map['network_type'] = network_type

        code, xml = self.submit(
            {'ambiente': network_map}, 'POST', 'environment/configuration/save/')

        return self.response(code, xml)