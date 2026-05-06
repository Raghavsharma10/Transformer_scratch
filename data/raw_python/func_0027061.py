def get_data(self, path):
        """Needs to return the string source for the module."""
        return LineCacheNotebookDecoder(
            code=self.code, raw=self.raw, markdown=self.markdown
        ).decode(self.decode(), self.path)