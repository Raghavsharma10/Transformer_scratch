def start_traffic(self, blocking=False, *ports):
        """ Start traffic on list of ports.

        :param blocking: True - start traffic and wait until traffic ends, False - start traffic and return.
        :param ports: list of ports to start traffic on. Default - all session ports.
        """

        for chassis, chassis_ports in self._per_chassis_ports(*self._get_operation_ports(*ports)).items():
            chassis.start_traffic(False, *chassis_ports)
        if blocking:
            for chassis, chassis_ports in self._per_chassis_ports(*self._get_operation_ports(*ports)).items():
                chassis.wait_traffic(*chassis_ports)