def _is_detach_necessary(cls):
        """Check if detaching the process is even necessary."""
        if os.getppid() == 1:
            # Process was started by init
            return False

        if cls._is_socket(sys.stdin):
            # If STDIN is a socket, the daemon was started by a super-server
            return False

        return True