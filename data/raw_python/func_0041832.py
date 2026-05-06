def unarchive(self, target_path=None, zip_path=None):
        """
        Extract the given files to the specified destination.

        :param src_path: The destination path where to extract the files.
        :type src_path: str
        :param zip_path: The file path of the ZIP archive.
        :type zip_path: str
        """
        if target_path:
            self.target_path = target_path

        if zip_path:
            self.zip_path = zip_path

        if self.has_path is False:
            raise RuntimeError("")

        if os.path.isdir(self.target_path) is False:
            os.mkdir(self.target_path)

        with zipfile.ZipFile(self.zip_path, 'r') as zip:
            zip.extractall(self.target_path)