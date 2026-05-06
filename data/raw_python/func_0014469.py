def _send_user_message(self, data):
        """
        send a message, but block if we're in key negotiation.  this is used
        for user-initiated requests.
        """
        start = time.time()
        while True:
            self.clear_to_send.wait(0.1)
            if not self.active:
                self._log(DEBUG, 'Dropping user packet because connection is dead.')
                return
            self.clear_to_send_lock.acquire()
            if self.clear_to_send.isSet():
                break
            self.clear_to_send_lock.release()
            if time.time() > start + self.clear_to_send_timeout:
              raise SSHException('Key-exchange timed out waiting for key negotiation')
        try:
            self._send_message(data)
        finally:
            self.clear_to_send_lock.release()