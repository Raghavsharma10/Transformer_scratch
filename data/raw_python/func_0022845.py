def parse(self):
        """Parse the lines, and fill self.line_fields accordingly."""
        for line in self.lines:
            # Parse the line
            field_defs = self.parse_line(line)
            fields = []

            # Convert field parameters into Field objects
            for (kind, options) in field_defs:
                logger.debug("Creating field %s(%r)", kind, options)
                fields.append(self.field_registry.create(kind, **options))

            # Add the list of Field objects to the 'fields per line'.
            self.line_fields.append(fields)

            # Pre-fill the list of widgets
            for field in fields:
                self.widgets[field] = None