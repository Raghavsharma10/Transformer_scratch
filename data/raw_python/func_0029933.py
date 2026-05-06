def clear_file_systems(self):
        """Remove references to build and source file systems, reverting to the defaults"""

        self._source_url = None
        self.dataset.config.library.source.url = None
        self._source_fs = None

        self._build_url = None
        self.dataset.config.library.build.url = None
        self._build_fs = None

        self.dataset.commit()