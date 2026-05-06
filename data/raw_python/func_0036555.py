def read(self):
        """Parse content and metadata of markdown files"""
        self.convertRSTmetaToMD()
        self._md = Markdown(extensions=['meta', 'codehilite(linenums=True)'])
        content = self._md.convert(self.source)
        metadata = self._parse_metadata(self._md.Meta)
        return content, metadata