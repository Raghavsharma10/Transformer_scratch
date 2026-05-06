def restart(self):
        """
        Performs a soft reload of the HAProxy process.
        """
        version = self.get_version()

        command = [
            "haproxy",
            "-f", self.config_file_path, "-p", self.pid_file_path
        ]
        if version and version >= (1, 5, 0):
            command.extend(["-L", self.peer.name])
        if os.path.exists(self.pid_file_path):
            with open(self.pid_file_path) as fd:
                command.extend(["-sf", fd.read().replace("\n", "")])

        try:
            output = subprocess.check_output(command)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to restart HAProxy: %s", str(e))
            return

        if output:
            logging.error("haproxy says: %s", output)

        logger.info("Gracefully restarted HAProxy.")