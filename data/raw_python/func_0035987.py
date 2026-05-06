def _is_socket(cls, stream):
        """Check if the given stream is a socket."""
        try:
            fd = stream.fileno()
        except ValueError:
            # If it has no file descriptor, it's not a socket
            return False

        sock = socket.fromfd(fd, socket.AF_INET, socket.SOCK_RAW)
        try:
            # This will raise a socket.error if it's not a socket
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
        except socket.error as ex:
            if ex.args[0] != errno.ENOTSOCK:
                # It must be a socket
                return True
        else:
            # If an exception wasn't raised, it's a socket
            return True