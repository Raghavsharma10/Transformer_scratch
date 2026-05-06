def close(self):
        """Close the connection to the email server."""
        try:
            try:
                self.connection.quit()
            except socket.sslerror:
                # This happens when calling quit() on a TLS connection
                # sometimes.
                self.connection.close()
            except Exception as e:
                logger.error(
                    "Error trying to close connection to server " "%s:%s: %s",
                    self.host,
                    self.port,
                    e,
                )
                if self.fail_silently:
                    return
                raise
        finally:
            self.connection = None