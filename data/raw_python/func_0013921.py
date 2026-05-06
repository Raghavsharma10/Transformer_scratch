def _flat_vports(self, connection_port):
        """Flat the virtual ports."""
        vports = []
        for vport in connection_port.virtual_ports:
            self._set_child_props(connection_port, vport)
            vports.append(vport)
        return vports