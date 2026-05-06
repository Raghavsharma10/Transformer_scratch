def check_keepalive(self):
        """Send keepalive/PING if necessary."""
        if self.sock != NC.INVALID_SOCKET and time.time() - self.last_msg_out >= self.keep_alive:
            if self.state == NC.CS_CONNECTED:
                self.send_pingreq()
            else:
                self.socket_close()