def finish(self):
        "End this session"
        log.debug("Session disconnected.")
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except: pass
        self.session_end()