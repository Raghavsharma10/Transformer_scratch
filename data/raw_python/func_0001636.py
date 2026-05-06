def _chunk_filter(self, extensions):
        """ Create a filter from the extensions and ignore files """
        if isinstance(extensions, six.string_types):
            extensions = extensions.split()

        def _filter(chunk):
            """ Exclusion filter """
            name = chunk['name']
            if extensions is not None:
                if not any(name.endswith(e) for e in extensions):
                    return False
            for pattern in self.state.ignore_re:
                if pattern.match(name):
                    return False
            for pattern in self.state.ignore:
                if fnmatch.fnmatchcase(name, pattern):
                    return False
            return True
        return _filter