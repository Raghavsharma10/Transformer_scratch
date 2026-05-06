def description(self):
        """Attribute that returns the plugin description from its docstring."""
        lines = []
        for line in self.__doc__.split('\n')[2:]:
            line = line.strip()
            if line:
                lines.append(line)
        return ' '.join(lines)