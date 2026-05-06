def export(self, location):
        """Export the svn repository at the url to the destination location"""
        url, rev = self.get_url_rev()
        logger.notify('Exporting svn repository %s to %s' % (url, location))
        logger.indent += 2
        try:
            if os.path.exists(location):
                # Subversion doesn't like to check out over an existing directory
                # --force fixes this, but was only added in svn 1.5
                rmtree(location)
            call_subprocess(
                [self.cmd, 'export', url, location],
                filter_stdout=self._filter, show_stdout=False)
        finally:
            logger.indent -= 2