def get_ip_on_network(self, network_name):
        """Given a network name, returns the IP address

        :param network_name: (str) Name of the network to search for
        :return: (str) IP address on the specified network or None
        """
        return self.get_scenario_host_ip_on_network(
            scenario_role_name=self.cons3rt_role_name,
            network_name=network_name
        )