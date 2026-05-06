def getPorts(self):
        """acquire ports to be used by the SC2 client launched by this process"""
        if self.ports: # no need to get ports if ports are al
            return self.ports
        if not self._gotPorts:
            self.ports = [
                portpicker.pick_unused_port(), # game_port
                portpicker.pick_unused_port(), # base_port
                portpicker.pick_unused_port(), # shared_port / init port
            ]
            self._gotPorts = True
        return self.ports