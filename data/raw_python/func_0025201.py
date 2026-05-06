def rewrite_properties(self, properties):
        """Set the properties and write to disk."""
        with self.__library_storage_lock:
            self.__library_storage = properties
        self.__write_properties(None)