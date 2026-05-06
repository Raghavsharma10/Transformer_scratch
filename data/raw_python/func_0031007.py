def build_path(self, source, path_patterns=None, strict=False,
                   domains=None):
        ''' Constructs a target filename for a file or dictionary of entities.

        Args:
            source (str, File, dict): The source data to use to construct the
                new file path. Must be one of:
                - A File object
                - A string giving the path of a File contained within the
                  current Layout.
                - A dict of entities, with entity names in keys and values in
                  values
            path_patterns (list): Optional path patterns to use to construct
                the new file path. If None, the Layout-defined patterns will
                be used.
            strict (bool): If True, all entities must be matched inside a
                pattern in order to be a valid match. If False, extra entities
                will be ignored so long as all mandatory entities are found.
            domains (str, list): Optional name(s) of domain(s) to scan for
                path patterns. If None, all domains are scanned. If two or more
                domains are provided, the order determines the precedence of
                path patterns (i.e., earlier domains will have higher
                precedence).
        '''

        if isinstance(source, six.string_types):
            if source not in self.files:
                source = join(self.root, source)

            source = self.get_file(source)

        if isinstance(source, File):
            source = source.entities

        if path_patterns is None:
            if domains is None:
                domains = list(self.domains.keys())
            path_patterns = []
            for dom in listify(domains):
                path_patterns.extend(self.domains[dom].path_patterns)

        return build_path(source, path_patterns, strict)