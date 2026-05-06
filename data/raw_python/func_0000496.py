def read_stats(self):
        """ Read current statistics from chassis.

        :return: dictionary {stream: {tx: {stat name: stat value}} rx: {tpld: {stat group {stat name: value}}}}
        """

        self.tx_statistics = TgnObjectsDict()
        for port in self.session.ports.values():
            for stream in port.streams.values():
                self.tx_statistics[stream] = stream.read_stats()

        tpld_statistics = XenaTpldsStats(self.session).read_stats()

        self.statistics = TgnObjectsDict()
        for stream, stream_stats in self.tx_statistics.items():
            self.statistics[stream] = OrderedDict()
            self.statistics[stream]['tx'] = stream_stats
            self.statistics[stream]['rx'] = OrderedDict()
            stream_tpld = stream.get_attribute('ps_tpldid')
            for tpld, tpld_stats in tpld_statistics.items():
                if tpld.id == stream_tpld:
                    self.statistics[stream]['rx'][tpld] = tpld_stats
        return self.statistics