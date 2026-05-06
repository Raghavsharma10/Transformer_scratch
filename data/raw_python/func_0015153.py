def socket_close(self):
        """Close our socket."""
        if self.sock != NC.INVALID_SOCKET:
            self.sock.close()
        self.sock = NC.INVALID_SOCKET