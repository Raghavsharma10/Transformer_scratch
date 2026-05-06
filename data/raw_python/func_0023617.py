def initialize_directories(self, root_dir):
        """Create all directories needed for logs and configs."""
        if not root_dir:
            root_dir = os.path.expanduser('~')

        # Create config directory, if it doesn't exist
        self.config_dir = os.path.join(root_dir, '.config/pueue')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)