def create_socket(self):
        """Create a socket for the daemon, depending on the directory location.

        Args:
            config_dir (str): The absolute path to the config directory used by the daemon.

        Returns:
            socket.socket: The daemon socket. Clients connect to this socket.

        """
        socket_path = os.path.join(self.config_dir, 'pueue.sock')
        # Create Socket and exit with 1, if socket can't be created
        try:
            if os.path.exists(socket_path):
                os.remove(socket_path)
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(socket_path)
            self.socket.setblocking(0)
            self.socket.listen(0)
            # Set file permissions
            os.chmod(socket_path, stat.S_IRWXU)
        except Exception:
            self.logger.error("Daemon couldn't socket. Aborting")
            self.logger.exception()
            sys.exit(1)

        return self.socket