def get_lines(self, config_access, visited_set):
        """
        get the lines for this section
        visited_set is used to avoid visiting same section
        twice, if we've got a diamond in the @is setup
        """
        if self in visited_set:
            return []
        lines = self.lines.copy()
        visited_set.add(self)
        for identity in self.identities:
            if config_access.get_keyfile(identity):
                lines.append(('IdentitiesOnly', ['yes']))
                lines.append(('IdentityFile', [pipes.quote(config_access.get_keyfile(identity))]))
        for section_name in self.types:
            section = config_access.get_section(section_name)
            lines += section.get_lines(config_access, visited_set)
        return lines