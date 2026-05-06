def build_source_files(self):
        """Return acessors to the build files"""

        from .files import BuildSourceFileAccessor
        return BuildSourceFileAccessor(self, self.dataset, self.source_fs)