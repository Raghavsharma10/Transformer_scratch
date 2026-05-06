def find_bound_ports(self, ports):
        """Find any ports that are already bound and complain about them"""
        bound = []
        for port in ports:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind((port.ip if port.ip is not NotSpecified else "127.0.0.1", port.host_port))
            except socket.error as error:
                bound.append(port.host_port)
            finally:
                s.close()

        if bound:
            raise AlreadyBoundPorts(ports=bound)