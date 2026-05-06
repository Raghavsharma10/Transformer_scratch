def detect_xid_devices(self):
        """
        For all of the com ports connected to the computer, send an
        XID command '_c1'.  If the device response with '_xid', it is
        an xid device.
        """
        self.__xid_cons = []

        for c in self.__com_ports:
            device_found = False
            for b in [115200, 19200, 9600, 57600, 38400]:
                con = XidConnection(c, b)

                try:
                    con.open()
                except SerialException:
                    continue

                con.flush_input()
                con.flush_output()
                returnval = con.send_xid_command("_c1", 5).decode('ASCII')

                if returnval.startswith('_xid'):
                    device_found = True
                    self.__xid_cons.append(con)

                    if(returnval != '_xid0'):
                        # set the device into XID mode
                        con.send_xid_command('c10')
                        con.flush_input()
                        con.flush_output()

                    # be sure to reset the timer to avoid the 4.66 hours
                    # problem. (refer to XidConnection.xid_input_found to
                    # read about the 4.66 hours)
                    con.send_xid_command('e1')
                    con.send_xid_command('e5')

                con.close()
                if device_found:
                    break