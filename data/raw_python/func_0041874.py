def format_epilog_section(self, section, text):
        """Format a section for the epilog by inserting a format"""
        try:
            func = self._epilog_formatters[self.epilog_formatter]
        except KeyError:
            if not callable(self.epilog_formatter):
                raise
            func = self.epilog_formatter
        return func(section, text)