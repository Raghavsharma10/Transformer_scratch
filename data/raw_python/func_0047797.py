def connect(self):
        """Create connection to CasparCG Server"""
        try:
            self.connection = telnetlib.Telnet(self.host, self.port, timeout=self.timeout)
        except Exception:
            log_traceback()
            return False
        return True