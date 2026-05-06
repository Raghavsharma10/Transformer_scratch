def clean(self):
        """Clear the contents of the build area."""
        if os.path.exists(self.buildroot):
            log.info('Clearing the build area.')
            log.debug('Deleting: %s', self.buildroot)
            shutil.rmtree(self.buildroot)
            os.makedirs(self.buildroot)