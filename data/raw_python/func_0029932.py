def set_file_system(self, source_url=False, build_url=False):
        """Set the source file filesystem and/or build  file system"""

        assert isinstance(source_url, string_types) or source_url is None or source_url is False
        assert isinstance(build_url, string_types) or build_url is False

        if source_url:
            self._source_url = source_url
            self.dataset.config.library.source.url = self._source_url
            self._source_fs = None

        elif source_url is None:
            self._source_url = None
            self.dataset.config.library.source.url = self._source_url
            self._source_fs = None

        if build_url:
            self._build_url = build_url
            self.dataset.config.library.build.url = self._build_url
            self._build_fs = None

        self.dataset.commit()