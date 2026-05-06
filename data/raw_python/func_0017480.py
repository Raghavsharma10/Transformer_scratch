def _send_command(self, command, expected_bytes):
        """
        Send an XID command to the device
        """
        response = self.con.send_xid_command(command, expected_bytes)

        return response