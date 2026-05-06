def perform(self):
        """
        Performs a straightforward TCP request and response.

        Sends the TCP `query` to the proper host and port, and loops over the
        socket, gathering response chunks until a full line is acquired.

        If the response line matches the expected value, the check passes. If
        not, the check fails.  The check will also fail if there's an error
        during any step of the send/receive process.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.connect((self.host, self.port))

        # if no query/response is defined, a successful connection is a pass
        if not self.query:
            sock.close()
            return True

        try:
            sock.sendall(self.query)
        except Exception:
            logger.exception("Error sending TCP query message.")
            sock.close()
            return False

        response, extra = sockutils.get_response(sock)

        logger.debug("response: %s (extra: %s)", response, extra)

        if response != self.expected_response:
            logger.warn(
                "Response does not match expected value: %s (expected %s)",
                response, self.expected_response
            )
            sock.close()
            return False

        sock.close()
        return True