def set_pulse_duration(self, duration):
        """
        Sets the pulse duration for events in miliseconds when activate_line
        is called
        """
        if duration > 4294967295:
            raise ValueError('Duration is too long. Please choose a value '
                             'less than 4294967296.')

        big_endian = hex(duration)[2:]
        if len(big_endian) % 2 != 0:
            big_endian = '0'+big_endian

        little_endian = []

        for i in range(0, len(big_endian), 2):
            little_endian.insert(0, big_endian[i:i+2])

        for i in range(0, 4-len(little_endian)):
            little_endian.append('00')

        command = 'mp'
        for i in little_endian:
            command += chr(int(i, 16))

        self.con.send_xid_command(command, 0)