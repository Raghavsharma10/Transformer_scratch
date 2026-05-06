def get_active_ports(self):
        """
        :return: dictionary {index: object} of all ports.
        """

        if not self.resource_groups:
            return self.ports
        else:
            active_ports = OrderedDict()
            for resource_group in self.resource_groups.values():
                for active_port in resource_group.active_ports:
                    active_ports[active_port] = self.ports[active_port]
            return active_ports