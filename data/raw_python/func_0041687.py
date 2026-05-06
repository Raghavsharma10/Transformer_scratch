def html(self, report, f):
        """
        Write report data to an HTML file.
        """
        env = Environment(loader=PackageLoader('clan', 'templates'))

        template = env.get_template('report.html')

        context = {
            'report': report,
            'GLOBAL_ARGUMENTS': GLOBAL_ARGUMENTS,
            'field_definitions': self.field_definitions,
            'format_comma': format_comma,
            'format_duration': format_duration,
            'format_percent': format_percent
        }

        f.write(template.render(**context).encode('utf-8'))