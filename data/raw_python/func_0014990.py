def disconnect(self):
        """
        Close the TCP connection with the graphite server.
        """
        try:
            self.socket.shutdown(1)

        # If its currently a socket, set it to None
        except AttributeError:
            self.socket = None
        except Exception:
            self.socket = None

        # Set the self.socket to None, no matter what.
        finally:
            self.socket = None