def _connect(self):
        """Do not call this directly - call auto_connect() or connect(),
        which will call _connect() for you.

        Connects to the roaster and creates communication thread.
        Raises a RoasterLokkupError exception if the hardware is not found.
        """
        # the following call raises a RoasterLookupException when the device
        # is not found. It is
        port = utils.find_device('1A86:5523')
        # on some systems, after the device port is added to the device list,
        # it can take up to 20 seconds after USB insertion for
        # the port to become available... (!)
        # let's put a safety timeout in here as a precaution
        wait_timeout = time.time() + 40.0  # should be PLENTY of time!
        # let's update the _connect_state while we're at it...
        self._connect_state.value = self.CS_CONNECTING
        connect_success = False
        while time.time() < wait_timeout:
            try:
                self._ser = serial.Serial(
                    port=port,
                    baudrate=9600,
                    bytesize=8,
                    parity='N',
                    stopbits=1.5,
                    timeout=0.25,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False)
                connect_success = True
                break
            except serial.SerialException:
                time.sleep(0.5)
        if not connect_success:
            # timeout on attempts
            raise exceptions.RoasterLookupError

        self._initialize()