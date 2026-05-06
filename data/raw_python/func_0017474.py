def query_base_timer(self):
        """
        gets the value from the device's base timer
        """
        (_, _, time) = unpack('<ccI', self.con.send_xid_command("e3", 6))
        return time