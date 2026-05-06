def parse(self, content):
        """
        Parse a stylesheet document with a regex (``REGEX_IMPORT_RULE``)
        to extract all import rules and return them.

        Args:
            content (str): A SCSS source.

        Returns:
            list: Finded paths in import rules.
        """
        # Remove all comments before searching for import rules, to not catch
        # commented breaked import rules
        declarations = self.REGEX_IMPORT_RULE.findall(
            self.remove_comments(content)
        )
        return self.flatten_rules(declarations)