def send_line(self, line, parse_result=True):
        """Submit a raw line of text to the VW instance, returning a 
        VWResult() object.

        If 'parse_result' is False, ignore the result and return None.
        """
        self.vw_process.sendline(line)  # Send line, along with newline
        result = self._get_response(parse_result=parse_result)
        return result