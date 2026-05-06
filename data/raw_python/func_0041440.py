def filename(self):
        """
        :return:
        :rtype: str
        """
        filename = self.key

        if self.has_revision_file() and self.history.current_revision:
            filename += "-"
            filename += self.history.current_revision.revision_id

        filename += ".zip"

        return filename