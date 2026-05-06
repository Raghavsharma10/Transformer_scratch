def strip_quotes(self, content):
        """
        Unquote given rule.

        Args:
            content (str): An import rule.

        Raises:
            InvalidImportRule: Raise exception if the rule is badly quoted
            (not started or not ended quotes).

        Returns:
            string: The given rule unquoted.
        """
        error_msg = "Following rule is badly quoted: {}"
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            return content[1:-1]
        # Quote starting but not ended
        elif (content.startswith('"') and not content.endswith('"')) or \
             (content.startswith("'") and not content.endswith("'")):
            raise InvalidImportRule(error_msg.format(content))
        # Quote ending but not started
        elif (not content.startswith('"') and content.endswith('"')) or \
             (not content.startswith("'") and content.endswith("'")):
            raise InvalidImportRule(error_msg.format(content))

        return content