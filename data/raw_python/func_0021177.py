def register_mappings(self, alias, package_name):
        """Register mappings from a package under given alias.

        :param alias: The alias.
        :param package_name: The package name.
        """
        # For backwards compatibility, we also allow for ES2 mappings to be
        # placed at the root level of the specified package path, and not in
        # the `<package-path>/v2` directory.
        if ES_VERSION[0] == 2:
            try:
                resource_listdir(package_name, 'v2')
                package_name += '.v2'
            except (OSError, IOError) as ex:
                if getattr(ex, 'errno', 0) != errno.ENOENT:
                    raise
                warnings.warn(
                    "Having mappings in a path which doesn't specify the "
                    "Elasticsearch version is deprecated. Please move your "
                    "mappings to a subfolder named according to the "
                    "Elasticsearch version which your mappings are intended "
                    "for. (e.g. '{}/v2/{}')".format(
                        package_name, alias),
                    PendingDeprecationWarning)
        else:
            package_name = '{}.v{}'.format(package_name, ES_VERSION[0])

        def _walk_dir(aliases, *parts):
            root_name = build_index_name(self.app, *parts)
            resource_name = os.path.join(*parts)

            if root_name not in aliases:
                self.number_of_indexes += 1

            data = aliases.get(root_name, {})

            for filename in resource_listdir(package_name, resource_name):
                index_name = build_index_name(
                    self.app,
                    *(parts + (filename, ))
                )
                file_path = os.path.join(resource_name, filename)

                if resource_isdir(package_name, file_path):
                    _walk_dir(data, *(parts + (filename, )))
                    continue

                ext = os.path.splitext(filename)[1]
                if ext not in {'.json', }:
                    continue

                assert index_name not in data, 'Duplicate index'
                data[index_name] = self.mappings[index_name] = \
                    resource_filename(
                        package_name, os.path.join(resource_name, filename))
                self.number_of_indexes += 1

            aliases[root_name] = data

        # Start the recursion here:
        _walk_dir(self.aliases, alias)