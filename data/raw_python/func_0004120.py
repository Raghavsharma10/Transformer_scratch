def add_handler(self, path, handler):
        """Add a path in watch queue
        """
        self.signatures[path] = self.get_path_signature(path)
        self.handlers[path] = handler