def get(self, return_type='tuple', target=None, extensions=None,
            domains=None, regex_search=None, **kwargs):
        """
        Retrieve files and/or metadata from the current Layout.

        Args:
            return_type (str): Type of result to return. Valid values:
                'tuple': returns a list of namedtuples containing file name as
                    well as attribute/value pairs for all named entities.
                'file': returns a list of matching filenames.
                'dir': returns a list of directories.
                'id': returns a list of unique IDs. Must be used together with
                    a valid target.
                'obj': returns a list of matching File objects.
            target (str): The name of the target entity to get results for
                (if return_type is 'dir' or 'id').
            extensions (str, list): One or more file extensions to filter on.
                Files with any other extensions will be excluded.
            domains (list): Optional list of domain names to scan for files.
                If None, all available domains are scanned.
            regex_search (bool or None): Whether to require exact matching
                (False) or regex search (True) when comparing the query string
                to each entity. If None (default), uses the value found in
                self.
            kwargs (dict): Any optional key/values to filter the entities on.
                Keys are entity names, values are regexes to filter on. For
                example, passing filter={ 'subject': 'sub-[12]'} would return
                only files that match the first two subjects.

        Returns:
            A named tuple (default) or a list (see return_type for details).
        """

        if regex_search is None:
            regex_search = self.regex_search

        result = []
        filters = {}
        filters.update(kwargs)

        for filename, file in self.files.items():
            if not file._matches(filters, extensions, domains, regex_search):
                continue
            result.append(file)

        # Convert to relative paths if needed
        if not self.absolute_paths:
            for i, f in enumerate(result):
                f = copy(f)
                f.path = relpath(f.path, self.root)
                result[i] = f

        if return_type == 'file':
            return natural_sort([f.path for f in result])

        if return_type == 'tuple':
            result = [r.as_named_tuple() for r in result]
            return natural_sort(result, field='filename')

        if return_type.startswith('obj'):
            return result

        else:
            valid_entities = self.get_domain_entities(domains)

            if target is None:
                raise ValueError('If return_type is "id" or "dir", a valid '
                                 'target entity must also be specified.')
            result = [x for x in result if target in x.entities]

            if return_type == 'id':
                result = list(set([x.entities[target] for x in result]))
                return natural_sort(result)

            elif return_type == 'dir':
                template = valid_entities[target].directory
                if template is None:
                    raise ValueError('Return type set to directory, but no '
                                     'directory template is defined for the '
                                     'target entity (\"%s\").' % target)
                # Construct regex search pattern from target directory template
                to_rep = re.findall('\{(.*?)\}', template)
                for ent in to_rep:
                    patt = valid_entities[ent].pattern
                    template = template.replace('{%s}' % ent, patt)
                template += '[^\%s]*$' % os.path.sep
                matches = [f.dirname for f in result
                           if re.search(template, f.dirname)]
                return natural_sort(list(set(matches)))

            else:
                raise ValueError("Invalid return_type specified (must be one "
                                 "of 'tuple', 'file', 'id', or 'dir'.")