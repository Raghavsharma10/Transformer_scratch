def flatten_rules(self, declarations):
        """
        Flatten returned import rules from regex.

        Because import rules can contains multiple items in the same rule
        (called multiline import rule), the regex ``REGEX_IMPORT_RULE``
        return a list of unquoted items for each rule.

        Args:
            declarations (list): A SCSS source.

        Returns:
            list: Given SCSS source with all comments removed.
        """
        rules = []

        for protocole, paths in declarations:
            # If there is a protocole (like 'url), drop it
            if protocole:
                continue
            # Unquote and possibly split multiple rule in the same declaration
            rules.extend([self.strip_quotes(v.strip())
                          for v in paths.split(',')])

        return list(filter(self.filter_rules, rules))