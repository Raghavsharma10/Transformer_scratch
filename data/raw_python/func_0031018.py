def read_stats(self, *stats):
        """ Read stream statistics from chassis.

        :param stats: list of requested statistics to read, if empty - read all statistics.
        """
        from ixexplorer.ixe_stream import IxePacketGroupStream
        sleep_time = 0.1  # in cases we only want few counters but very fast we need a smaller sleep time
        if not stats:
            stats = [m.attrname for m in IxePgStats.__tcl_members__ if m.flags & FLAG_RDONLY]
            sleep_time = 1

        # Read twice to refresh rate statistics.
        for port in self.tx_ports_streams:
            port.api.call_rc('streamTransmitStats get {} 1 4096'.format(port.uri))
        for rx_port in self.rx_ports:
            rx_port.api.call_rc('packetGroupStats get {} 0 65536'.format(rx_port.uri))
        time.sleep(sleep_time)

        self.statistics = OrderedDict()
        for tx_port, streams in self.tx_ports_streams.items():
            for stream in streams:
                stream_stats = OrderedDict()
                tx_port.api.call_rc('streamTransmitStats get {} 1 4096'.format(tx_port.uri))
                stream_tx_stats = IxeStreamTxStats(tx_port, stream.index)
                stream_stats_tx = {c: v for c, v in stream_tx_stats.get_attributes(FLAG_RDONLY).items()}
                stream_stats['tx'] = stream_stats_tx
                stream_stat_pgid = IxePacketGroupStream(stream).groupId
                stream_stats_pg = pg_stats_dict()
                for port in self.session.ports.values():
                    stream_stats_pg[str(port)] = OrderedDict(zip(stats, [-1] * len(stats)))
                for rx_port in self.rx_ports:
                    if not stream.rx_ports or rx_port in stream.rx_ports:
                        rx_port.api.call_rc('packetGroupStats get {} 0 65536'.format(rx_port.uri))
                        pg_stats = IxePgStats(rx_port, stream_stat_pgid)
                        stream_stats_pg[str(rx_port)] = pg_stats.read_stats(*stats)
                stream_stats['rx'] = stream_stats_pg
                self.statistics[str(stream)] = stream_stats
        return self.statistics