def read_stats(self, *stats):
        """ Read port statistics from chassis.

        :param stats: list of requested statistics to read, if empty - read all statistics.
        """

        self.statistics = OrderedDict()
        for port in self.ports:
            port_stats = IxeStatTotal(port).get_attributes(FLAG_RDONLY, *stats)
            port_stats.update({c + '_rate': v for c, v in
                               IxeStatRate(port).get_attributes(FLAG_RDONLY, *stats).items()})
            self.statistics[str(port)] = port_stats
        return self.statistics