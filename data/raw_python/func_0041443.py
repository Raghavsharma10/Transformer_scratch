def infofile_path(self):
        """
        :return:
        :rtype: str
        """
        return os.path.normpath(os.path.join(
            self.dest_path,
            self.config.info_file
        ))