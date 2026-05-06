def start(self):
        """
        Start downloading, handling auto retry, download resume and path
        moving
        """
        if not self.auto_retry:
            self.curl()
            return
        while not self.is_finished:
            try:
                self.curl()
            except pycurl.error as e:
                # transfer closed with n bytes remaining to read
                if e.args[0] == pycurl.E_PARTIAL_FILE:
                    pass
                # HTTP server doesn't seem to support byte ranges.
                # Cannot resume.
                elif e.args[0] == pycurl.E_HTTP_RANGE_ERROR:
                    break
                # Recv failure: Connection reset by peer
                elif e.args[0] == pycurl.E_RECV_ERROR:
                    if self._rst_retries < self.max_rst_retries:
                        pass
                    else:
                        raise e
                    self._rst_retries += 1
                else:
                    raise e
        self._move_path()
        self._done()