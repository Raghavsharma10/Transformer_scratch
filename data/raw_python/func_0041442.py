def revfile_path(self):
        """
        :return: The full path of revision file.
        :rtype: str
        """
        return os.path.normpath(os.path.join(
            os.getcwd(),
            self.config.revision_file
        ))