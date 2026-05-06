def receive_forever(self):
        """
        Receive and handle messages in an endless loop. A message may consist
        of multiple data frames, but this is not visible for onmessage().
        Control messages (or control frames) are handled automatically.
        """
        while True:
            try:
                self.onmessage(self.recv())
            except (KeyboardInterrupt, SystemExit, SocketClosed):
                break
            except Exception as e:
                self.onerror(e)
                self.onclose(None, 'error: %s' % e)

                try:
                    self.sock.close()
                except socket.error:
                    pass

                raise e