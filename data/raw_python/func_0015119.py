def loop_misc(self):
        """Misc loop."""
        self.check_keepalive()
        if self.last_retry_check + 1  < time.time():
            pass
        return NC.ERR_SUCCESS