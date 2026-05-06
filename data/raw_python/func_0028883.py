def _write_pidfile(self):
        """Write the pid file out with the process number in the pid file"""
        LOGGER.debug('Writing pidfile: %s', self.pidfile_path)
        with open(self.pidfile_path, "w") as handle:
            handle.write(str(os.getpid()))