def documentation(self):
        """Return the documentation, from the documentation.md file, with template substitutions"""

        # Return the documentation as a scalar term, which has .text() and .html methods to do
        # metadata substitution using Jinja

        s = ''

        rc = self.build_source_files.documentation.record_content

        if rc:
            s += rc

        for k, v in  self.metadata.documentation.items():
            if  v:
                s += '\n### {}\n{}'.format(k.title(), v)

        return self.metadata.scalar_term(s)