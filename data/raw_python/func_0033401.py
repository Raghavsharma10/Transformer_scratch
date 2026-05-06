def _get_response(self, parse_result=True):
        """If 'parse_result' is False, ignore the received output and return None."""
        # expect_exact is faster than just exact, and fine for our purpose
        # (http://pexpect.readthedocs.org/en/latest/api/pexpect.html#pexpect.spawn.expect_exact)
        # searchwindowsize and other attributes may also affect efficiency
        self.vw_process.expect_exact('\r\n', searchwindowsize=-1)  # Wait until process outputs a complete line
        if parse_result:
            output = self.vw_process.before
            result_struct = VWResult(output, active_mode=self.active_mode)
        else:
            result_struct = None
        return result_struct