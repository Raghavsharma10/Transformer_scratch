def _num_plurals(self, catalogue):
        """
        Return the number of plurals for this catalog language, or 2 if no
        plural string is available.
        """
        match = re.search(r'nplurals=\s*(\d+)', self.get_plural(catalogue) or '')
        if match:
            return int(match.groups()[0])
        return 2