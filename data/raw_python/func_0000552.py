def start_traffic(self, blocking=False, *ports):
        """ Start traffic on list of ports.

        :param blocking: True - start traffic and wait until traffic ends, False - start traffic and return.
        :param ports: list of ports to start traffic on. Default - all session ports.
        """

        self._traffic_command('on', *ports)
        if blocking:
            self.wait_traffic(*ports)