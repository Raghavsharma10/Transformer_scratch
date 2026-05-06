def register_schemas_dir(self, directory):
        """Recursively register all json-schemas in a directory.

        :param directory: directory path.
        """
        for root, dirs, files in os.walk(directory):
            dir_path = os.path.relpath(root, directory)
            if dir_path == '.':
                dir_path = ''
            for file_ in files:
                if file_.lower().endswith(('.json')):
                    schema_name = os.path.join(dir_path, file_)
                    if schema_name in self.schemas:
                        raise JSONSchemaDuplicate(
                            schema_name,
                            self.schemas[schema_name],
                            directory
                        )
                    self.schemas[schema_name] = os.path.abspath(directory)