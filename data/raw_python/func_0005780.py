def replace(self, match, content):
        """Replace all occurences of the regex in all matches
        from a file with a specific value.
        """
        new_string = self.replace_expression.sub(self.replace_with, match)
        logger.info('Replacing: [ %s ] --> [ %s ]', match, new_string)
        new_content = content.replace(match, new_string)
        return new_content