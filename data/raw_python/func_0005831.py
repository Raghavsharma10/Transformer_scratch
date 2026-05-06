def iter_options(self):
        """Iterates configuration sections groups options."""
        for section in self.sections:
            name = str(section)
            for key, value in section._get_options():
                yield name, key, value