def tmp_file_path(self):
        """
        :return:
        :rtype: str
        """
        return os.path.normpath(os.path.join(
            TMP_DIR,
            self.filename
        ))