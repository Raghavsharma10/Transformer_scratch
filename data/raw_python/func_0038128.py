def content(self):
        """
        Returns raw CSV content of the log file.
        """
        raw_content = self._manager.api.session.get(self.download_link).content
        data = BytesIO(raw_content)
        archive = ZipFile(data)
        filename = archive.filelist[0]  # Always 1 file in the archive
        return archive.read(filename)