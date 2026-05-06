def _remove_pidfile(self):
        """Remove the pid file from the filesystem"""
        LOGGER.debug('Removing pidfile: %s', self.pidfile_path)
        try:
            os.unlink(self.pidfile_path)
        except OSError:
            pass