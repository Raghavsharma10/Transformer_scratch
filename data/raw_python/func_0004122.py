def check(self):
        """Check if a file is changed
        """
        for (path, handler) in self.handlers.items():
            current_signature = self.signatures[path]
            new_signature = self.get_path_signature(path)
            if new_signature != current_signature:
                self.signatures[path] = new_signature
                handler.on_change(Event(path))