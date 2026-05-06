def check(self):
        """Checks whether loaded yaml is well-formed according to syntax defined for
        version 0.9.0 and later.

        Raises:
            YamlError: (containing a meaningful message) when the loaded Yaml
                is not well formed
        """
        if not isinstance(self.parsed_yaml, dict):
            msg = 'In {0}:\n'.format(self.sourcefile)
            msg += 'Assistants and snippets must be Yaml mappings, not "{0}"!'.\
                format(self.parsed_yaml)
            raise exceptions.YamlTypeError(msg)
        self._check_fullname(self.sourcefile)
        self._check_description(self.sourcefile)
        self._check_section_names(self.sourcefile)
        self._check_project_type(self.sourcefile)
        self._check_args(self.sourcefile)
        self._check_files(self.sourcefile)
        self._check_dependencies(self.sourcefile)
        self._check_run(self.sourcefile)