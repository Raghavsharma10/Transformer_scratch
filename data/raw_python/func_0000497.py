def read_stats(self):
        """ Read current statistics from chassis.

        :return: dictionary {tpld full index {group name {stat name: stat value}}}
        """

        self.statistics = TgnObjectsDict()
        for port in self.session.ports.values():
            for tpld in port.tplds.values():
                self.statistics[tpld] = tpld.read_stats()
        return self.statistics