def expand(self, repex_vars, fields):
        r"""Receive a dict of variables and a dict of fields
        and iterates through them to expand a variable in an field, then
        returns the fields dict with its variables expanded.

        This will fail if not all variables expand (due to not providing
        all necessary ones).

        fields:

        type: VERSION
        path: resources
        excluded:
            - excluded_file.file
        base_directory: '{{ .base_dir }}'
        match: '"version": "\d+\.\d+(\.\d+)?(-\w\d+)?'
        replace: \d+\.\d+(\.\d+)?(-\w\d+)?
        with: "{{ .version }}"
        must_include:
            - {{ .my_var }}/{{ .another_var }}
            - {{ .my_other_var }}
            - version
        validator:
            type: per_file
            path: {{ .my_validator_path }}
            function: validate

        variables:

        {
            'version': 3,
            'base_dir': .
            ...
        }

        :param dict vars: dict of variables
        :param dict fields: dict of fields as shown above.
        """
        logger.debug('Expanding variables...')

        unexpanded_instances = set()

        # Expand variables in variables
        # TODO: This should be done in the global scope.
        # _VariableHandler is called per path, which makes this redundant
        # as variables are declared globally per config.
        for k, v in repex_vars.items():
            repex_vars[k] = self._expand_var(v, repex_vars)
            instances = self._get_instances(repex_vars[k])
            unexpanded_instances.update(instances)

        # TODO: Consolidate variable expansion code into single logic
        # Expand variables in path objects
        for key in fields.keys():
            field = fields[key]
            if isinstance(field, str):
                fields[key] = self._expand_var(field, repex_vars)
                instances = self._get_instances(fields[key])
                unexpanded_instances.update(instances)
            elif isinstance(field, dict):
                for k, v in field.items():
                    fields[key][k] = self._expand_var(v, repex_vars)
                    instances = self._get_instances(fields[key][k])
                    unexpanded_instances.update(instances)
            elif isinstance(field, list):
                for index, item in enumerate(field):
                    fields[key][index] = self._expand_var(item, repex_vars)
                    instances = self._get_instances(fields[key][index])
                    unexpanded_instances.update(instances)

        if unexpanded_instances:
            raise RepexError(
                'Variables failed to expand: {0}\n'
                'Please make sure to provide all necessary variables '.format(
                    list(unexpanded_instances)))

        return fields