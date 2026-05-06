def format_help_text(self, ctx, formatter):
        """Writes the help text to the formatter if it exists."""
        if self.help:
            formatter.write_paragraph()
            with formatter.indentation():
                formatter.write_text(self.help)