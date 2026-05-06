def register_templates(self, directory):
        """Register templates from the provided directory.

        :param directory: The templates directory.
        """
        try:
            resource_listdir(directory, 'v{}'.format(ES_VERSION[0]))
            directory = '{}/v{}'.format(directory, ES_VERSION[0])
        except (OSError, IOError) as ex:
            if getattr(ex, 'errno', 0) == errno.ENOENT:
                raise OSError(
                    "Please move your templates to a subfolder named "
                    "according to the Elasticsearch version "
                    "which your templates are intended "
                    "for. (e.g. '{}.v{}')".format(directory,
                                                  ES_VERSION[0]))
        result = {}
        module_name, parts = directory.split('.')[0], directory.split('.')[1:]
        parts = tuple(parts)

        def _walk_dir(parts):
            resource_name = os.path.join(*parts)

            for filename in resource_listdir(module_name, resource_name):
                template_name = build_index_name(
                    self.app,
                    *(parts[1:] + (filename, ))
                )
                file_path = os.path.join(resource_name, filename)

                if resource_isdir(module_name, file_path):
                    _walk_dir((parts + (filename, )))
                    continue

                ext = os.path.splitext(filename)[1]
                if ext not in {'.json', }:
                    continue

                result[template_name] = resource_filename(
                    module_name, os.path.join(resource_name, filename))

        # Start the recursion here:
        _walk_dir(parts)
        return result