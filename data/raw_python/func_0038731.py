def render_source(self):
        """Render the sourcecode."""
        return SOURCE_TABLE_HTML % text_('\n'.join(line.render() for line in
            self.get_annotated_lines()))