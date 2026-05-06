def get_nearest(self, path, return_type='file', strict=True, all_=False,
                    ignore_strict_entities=None, full_search=False, **kwargs):
        ''' Walk up the file tree from the specified path and return the
        nearest matching file(s).

        Args:
            path (str): The file to search from.
            return_type (str): What to return; must be one of 'file' (default)
                or 'tuple'.
            strict (bool): When True, all entities present in both the input
                path and the target file(s) must match perfectly. When False,
                files will be ordered by the number of matching entities, and
                partial matches will be allowed.
            all_ (bool): When True, returns all matching files. When False
                (default), only returns the first match.
            ignore_strict_entities (list): Optional list of entities to
                exclude from strict matching when strict is True. This allows
                one to search, e.g., for files of a different type while
                matching all other entities perfectly by passing
                ignore_strict_entities=['type'].
            full_search (bool): If True, searches all indexed files, even if
                they don't share a common root with the provided path. If
                False, only files that share a common root will be scanned.
            kwargs: Optional keywords to pass on to .get().
        '''

        entities = {}
        for ent in self.entities.values():
            m = ent.regex.search(path)
            if m:
                entities[ent.name] = ent._astype(m.group(1))

        # Remove any entities we want to ignore when strict matching is on
        if strict and ignore_strict_entities is not None:
            for k in ignore_strict_entities:
                entities.pop(k, None)

        results = self.get(return_type='file', **kwargs)

        folders = defaultdict(list)

        for filename in results:
            f = self.get_file(filename)
            folders[f.dirname].append(f)

        def count_matches(f):
            f_ents = f.entities
            keys = set(entities.keys()) & set(f_ents.keys())
            shared = len(keys)
            return [shared, sum([entities[k] == f_ents[k] for k in keys])]

        matches = []

        search_paths = []
        while True:
            if path in folders and folders[path]:
                search_paths.append(path)
            parent = dirname(path)
            if parent == path:
                break
            path = parent

        if full_search:
            unchecked = set(folders.keys()) - set(search_paths)
            search_paths.extend(path for path in unchecked if folders[path])

        for path in search_paths:
            # Sort by number of matching entities. Also store number of
            # common entities, for filtering when strict=True.
            num_ents = [[f] + count_matches(f) for f in folders[path]]
            # Filter out imperfect matches (i.e., where number of common
            # entities does not equal number of matching entities).
            if strict:
                num_ents = [f for f in num_ents if f[1] == f[2]]
            num_ents.sort(key=lambda x: x[2], reverse=True)

            if num_ents:
                matches.append(num_ents[0][0])

            if not all_:
                break

        matches = [m.path if return_type == 'file' else m.as_named_tuple()
                   for m in matches]
        return matches if all_ else matches[0] if matches else None