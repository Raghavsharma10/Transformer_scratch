def remove_file(self, path):
        """Removes the given file"""
        self.get_file(path).remove()
        self.remove_cache_buster(path)