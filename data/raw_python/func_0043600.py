def add_part(self, part):
        """
        Function for adding partial pattern to the value
        :param part: string or compiled pattern
        """
        if isinstance(part, RE_TYPE):
            part = part.pattern

        # Allow U / spmething syntax
        if self == '^$':
            return URLPattern(part, self.separator)
        else:
            # Erase dup separator inbetween
            sep = self.separator
            return URLPattern(self.rstrip('$' + sep) + sep + part.lstrip(sep),
                              sep)