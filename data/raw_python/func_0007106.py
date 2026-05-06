def base_recherche_rapide(self, base, pattern, to_string_hook=None):
        """
        Return a collection of access matching `pattern`.
        `to_string_hook` is an optionnal callable dict -> str to map record to string. Default to _record_to_string
        """
        Ac = self.ACCES
        if pattern == "*":
            return groups.Collection(Ac(base, i) for i in self)

        if len(pattern) >= MIN_CHAR_SEARCH:  # Needed chars.
            sub_patterns = pattern.split(" ")
            try:
                regexps = tuple(re.compile(sub_pattern, flags=re.I)
                                for sub_pattern in sub_patterns)
            except re.error:
                return groups.Collection()

            def search(string):
                for regexp in regexps:
                    if not regexp.search(string):
                        return False
                return True

            to_string_hook = to_string_hook or self._record_to_string
            return groups.Collection(Ac(base, i) for i, p in self.items() if search(to_string_hook(p)))

        return groups.Collection()