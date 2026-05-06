def replace_placeholders(self, value):
        """Replaces placeholders that can be used e.g. in filepaths.

        Supported placeholders:
            * {project_runtime_dir}
            * {project_name}
            * {runtime_dir}

        :param str|unicode|list[str|unicode]|None value:
        :rtype: None|str|unicode|list[str|unicode]

        """
        if not value:
            return value

        is_list = isinstance(value, list)
        values = []

        for value in listify(value):
            runtime_dir = self.get_runtime_dir()
            project_name = self.project_name

            value = value.replace('{runtime_dir}', runtime_dir)
            value = value.replace('{project_name}', project_name)
            value = value.replace('{project_runtime_dir}', os.path.join(runtime_dir, project_name))

            values.append(value)

        value = values if is_list else values.pop()

        return value