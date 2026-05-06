def daemon_start(self):
        """Start daemon when gtk loaded
        """
        if daemon_status() == "SUN not running":
            subprocess.call("{0} &".format(self.cmd), shell=True)