def delete_file(self, path, prefixed_path, source_storage):
        """
        We don't need all the file_exists stuff because we have to override all files anyways.
        """
        if self.faster:
            return True
        else:
            return super(Command, self).delete_file(path, prefixed_path, source_storage)