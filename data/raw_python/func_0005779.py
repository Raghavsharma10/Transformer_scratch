def find_matches(self, content, file_to_handle):
        """Find all matches of an expression in a file
        """
        # look for all match groups in the content
        groups = [match.groupdict() for match in
                  self.match_expression.finditer(content)]
        # filter out content not in the matchgroup
        matches = [group['matchgroup'] for group in groups
                   if group.get('matchgroup')]

        logger.info('Found %s matches in %s', len(matches), file_to_handle)
        # We only need the unique strings found as we'll be replacing each
        # of them. No need to replace the ones already replaced.
        return list(set(matches))