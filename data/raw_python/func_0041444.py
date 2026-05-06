def dest_path(self):
        """
        :return: The destination path.
        :rtype: str
        """
        if os.path.isabs(self.config.local_path):
            return self.config.local_path
        else:
            return os.path.normpath(os.path.join(
                os.getcwd(),
                self.config.local_path
            ))