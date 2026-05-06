def get_cap_files(self, *ports):
        """
        :param ports: list of ports to get capture files names for.
        :return: dictionary (port, capture file)
        """
        cap_files = {}
        for port in ports:
            if port.cap_file_name:
                with open(port.cap_file_name) as f:
                    cap_files[port] = f.read().splitlines()
            else:
                cap_files[port] = None
        return cap_files