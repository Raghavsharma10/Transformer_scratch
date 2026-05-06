def set_file_path(self, path):
        """Update the file_path Entry widget"""
        self.file_path.delete(0, END)
        self.file_path.insert(0, path)