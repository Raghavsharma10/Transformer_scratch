def opened(self, *args):
        """Initiates communication with the remote controlled device.

        :param args:
        """
        self._serial_open = True

        self.log("Opened: ", args, lvl=debug)
        self._send_command(b'l,1')  # Saying hello, shortly
        self.log("Turning off engine, pump and neutralizing rudder")
        self._send_command(b'v')
        self._handle_servo(self._machine_channel, 0)
        self._handle_servo(self._rudder_channel, 127)
        self._set_digital_pin(self._pump_channel, 0)
        # self._send_command(b'h')
        self._send_command(b'l,0')
        self._send_command(b'm,HFOS Control')