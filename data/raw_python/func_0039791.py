def get_info(self, location):
        """Returns (url, revision), where both are strings"""
        assert not location.rstrip('/').endswith(self.dirname), 'Bad directory: %s' % location
        output = call_subprocess(
            [self.cmd, 'info', location], show_stdout=False, extra_environ={'LANG': 'C'})
        match = _svn_url_re.search(output)
        if not match:
            logger.warn('Cannot determine URL of svn checkout %s' % display_path(location))
            logger.info('Output that cannot be parsed: \n%s' % output)
            return None, None
        url = match.group(1).strip()
        match = _svn_revision_re.search(output)
        if not match:
            logger.warn('Cannot determine revision of svn checkout %s' % display_path(location))
            logger.info('Output that cannot be parsed: \n%s' % output)
            return url, None
        return url, match.group(1)