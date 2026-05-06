def remove_temporary_source(self):
        """Remove the source files from this requirement, if they are marked
        for deletion"""
        if self.is_bundle or os.path.exists(self.delete_marker_filename):
            logger.info('Removing source in %s' % self.source_dir)
            if self.source_dir:
                rmtree(self.source_dir)
            self.source_dir = None
        if self._temp_build_dir and os.path.exists(self._temp_build_dir):
            rmtree(self._temp_build_dir)
        self._temp_build_dir = None