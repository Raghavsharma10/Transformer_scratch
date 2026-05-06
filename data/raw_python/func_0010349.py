def _writer(self):
        """
        Indefinitely checks the writer queue for data to write
        to socket.
        """
        while not self.closed:
            try:
                sock, data = self._write_queue.get(timeout=0.1)
                self._write_queue.task_done()
                sock.send(data)
            except Empty:
                pass  # nothing to write after timeout
            except socket.error as err:
                if err.errno == errno.EBADF:
                    self._clean_dead_sessions()