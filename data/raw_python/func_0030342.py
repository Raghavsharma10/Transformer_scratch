def get(name, default=None):
        """
        Return variable by name from the project's config.

        Name can be a dotted path, like: 'rails.db.type'.
        """
        if '.' not in name:
            raise Exception("Config path should be divided by at least one dot")
        section_name, var_path = name.split('.', 1)
        section = Config._data.get(section_name)
        return section.get(var_path)