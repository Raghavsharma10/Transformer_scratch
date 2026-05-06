def read_stats(self):
        """ Read current ports statistics from chassis.

        :return: dictionary {port name {group name, {stat name: stat value}}}
        """

        self.statistics = TgnObjectsDict()
        for port in self.session.ports.values():
            self.statistics[port] = port.read_port_stats()
        return self.statistics