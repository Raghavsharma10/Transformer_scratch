def set_path(self, file_path):
        """
        Set the path of the database.
        Create the file if it does not exist.
        """
        if not file_path:
            self.read_data = self.memory_read
            self.write_data = self.memory_write
        elif not is_valid(file_path):
            self.write_data(file_path, {})

        self.path = file_path