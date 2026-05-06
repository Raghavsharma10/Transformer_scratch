def set_server_dir(self, dir):
        """
        Set the directory of the server to be controlled
        """
        self.dir = os.path.abspath(dir)
        config = os.path.join(self.dir, 'etc', 'grid', 'config.xml')
        self.configured = os.path.exists(config)