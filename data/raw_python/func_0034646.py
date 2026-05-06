def render_xsl(self, node, context):
        """Render all XSL elements"""

        for e in self.xsl_elements:
            e.render(e.run)