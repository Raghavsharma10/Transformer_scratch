def validate_before(self, content, file_to_handle):
        """Verify that all required strings are in the file
        """
        logger.debug('Looking for required strings: %s', self.must_include)
        included = True
        for string in self.must_include:
            if not re.search(r'{0}'.format(string), content):
                logger.error('Required string `%s` not found in %s',
                             string, file_to_handle)
                included = False
        if not included:
            logger.debug('Required strings not found')
            return False
        logger.debug('Required strings found')
        return True