def _default_interface(self, route_output=None):
        """
        :param route_output: For mocking actual output
        """
        if not route_output:
            out, __, __ = exec_cmd('/sbin/ip route')
            lines = out.splitlines()
        else:
            lines = route_output.split("\n")

        for line in lines:
            line = line.split()
            if 'default' in line:
                iface = line[4]
                return self.interfaces.get(iface, None)