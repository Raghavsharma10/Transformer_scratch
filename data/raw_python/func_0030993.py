def _matches(self, entities=None, extensions=None, domains=None,
                 regex_search=False):
        """
        Checks whether the file matches all of the passed entities and
        extensions.

        Args:
            entities (dict): A dictionary of entity names -> regex patterns.
            extensions (str, list): One or more file extensions to allow.
            domains (str, list): One or more domains the file must match.
            regex_search (bool): Whether to require exact match (False) or
                regex search (True) when comparing the query string to each
                entity.
        Returns:
            True if _all_ entities and extensions match; False otherwise.
        """
        if extensions is not None:
            if isinstance(extensions, six.string_types):
                extensions = [extensions]
            extensions = '(' + '|'.join(extensions) + ')$'
            if re.search(extensions, self.filename) is None:
                return False

        if domains is not None:
            domains = listify(domains)
            if not set(self.domains) & set(domains):
                return False

        if entities is not None:

            for name, val in entities.items():

                if (name not in self.tags) ^ (val is None):
                    return False

                if val is None:
                    continue

                def make_patt(x):
                    patt = '%s' % x
                    if isinstance(x, (int, float)):
                        # allow for leading zeros if a number was specified
                        # regardless of regex_search
                        patt = '0*' + patt
                    if not regex_search:
                        patt = '^%s$' % patt
                    return patt

                ent_patts = [make_patt(x) for x in listify(val)]
                patt = '|'.join(ent_patts)

                if re.search(patt, str(self.tags[name].value)) is None:
                    return False
        return True