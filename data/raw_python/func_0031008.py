def write_contents_to_file(self, entities, path_patterns=None,
                               contents=None, link_to=None,
                               content_mode='text', conflicts='fail',
                               strict=False, domains=None, index=False,
                               index_domains=None):
        """
        Write arbitrary data to a file defined by the passed entities and
        path patterns.

        Args:
            entities (dict): A dictionary of entities, with Entity names in
                keys and values for the desired file in values.
            path_patterns (list): Optional path patterns to use when building
                the filename. If None, the Layout-defined patterns will be
                used.
            contents (object): Contents to write to the generate file path.
                Can be any object serializable as text or binary data (as
                defined in the content_mode argument).
            conflicts (str): One of 'fail', 'skip', 'overwrite', or 'append'
            that defines the desired action when the output path already
            exists. 'fail' raises an exception; 'skip' does nothing;
            'overwrite' overwrites the existing file; 'append' adds a suffix
            to each file copy, starting with 1. Default is 'fail'.
            strict (bool): If True, all entities must be matched inside a
                pattern in order to be a valid match. If False, extra entities
                will be ignored so long as all mandatory entities are found.
            domains (list): List of Domains to scan for path_patterns. Order
                determines precedence (i.e., earlier Domains will be scanned
                first). If None, all available domains are included.
            index (bool): If True, adds the generated file to the current
                index using the domains specified in index_domains.
            index_domains (list): List of domain names to attach the generated
                file to when indexing. Ignored if index == False.  If None,
                All available domains are used.

        """
        path = self.build_path(entities, path_patterns, strict, domains)

        if path is None:
            raise ValueError("Cannot construct any valid filename for "
                             "the passed entities given available path "
                             "patterns.")

        write_contents_to_file(path, contents=contents, link_to=link_to,
                               content_mode=content_mode, conflicts=conflicts,
                               root=self.root)

        if index:
            # TODO: Default to using only domains that have at least one
            # tagged entity in the generated file.
            if index_domains is None:
                index_domains = list(self.domains.keys())
            self._index_file(self.root, path, index_domains)