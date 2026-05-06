def available_ports():
        """
        Scans COM1 through COM255 for available serial ports

        returns a list of available ports
        """
        ports = []

        for i in range(256):
            try:
                p = Serial('COM%d' % i)
                p.close()
                ports.append(p)
            except SerialException:
                pass

        return ports