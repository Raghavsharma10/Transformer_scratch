def _substitute_file_uuids_throughout_template(self, template, file_dependencies):
        """Anywhere in "template" that refers to a data object but does not 
        give a specific UUID, if a matching file can be found in "file_dependencies",
        we will change the data object reference to use that UUID. That way templates
        have a preference to connect to files nested under their ".dependencies" over
        files that were previously imported to the server.
        """
        if not isinstance(template, dict):
            # Nothing to do if this is a reference to a previously imported template.
            return
        for input in template.get('inputs', []):
            self._substitute_file_uuids_in_input(input, file_dependencies)
        for step in template.get('steps', []):
            self._substitute_file_uuids_throughout_template(step, file_dependencies)