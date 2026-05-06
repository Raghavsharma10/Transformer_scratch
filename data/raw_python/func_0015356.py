def get_assistants(cls, superassistants):
        """Returns list of assistants that are subassistants of given superassistants
        (I love this docstring).

        Args:
            roles: list of names of roles, defaults to all roles
        Returns:
            list of YamlAssistant instances with specified roles
        """
        _assistants = cls.load_all_assistants(superassistants)
        result = []
        for supa in superassistants:
            result.extend(_assistants[supa.name])

        return result