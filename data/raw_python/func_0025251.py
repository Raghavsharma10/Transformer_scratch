def rewrite_properties(self, properties):
        """Set the properties and write to disk."""
        with self.__properties_lock:
            self.__properties = properties
        self.__write_properties(None)