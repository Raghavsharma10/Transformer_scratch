def txt(self, diff, f):
        """
        Generate a text report for a diff.
        """
        env = Environment(
            loader=PackageLoader('clan', 'templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )

        template = env.get_template('diff.txt')

        def format_row(label, values):
            change = format_comma(values['change'])
            percent_change = '{:.1%}'.format(values['percent_change']) if values['percent_change'] is not None else '-'
            point_change = '{:.1f}'.format(values['point_change'] * 100) if values['point_change'] is not None else '-'

            if values['change'] > 0:
                change = '+%s' % change

            if values['percent_change'] is not None and values['percent_change'] > 0:
                percent_change = '+%s' % percent_change

            if values['point_change'] is not None and values['point_change'] > 0:
                point_change = '+%s' % point_change

            return '{:>15s}    {:>8s}    {:>8s}    {:s}\n'.format(change, percent_change, point_change, label)

        context = {
            'diff': diff,
            'field_definitions': self.field_definitions,
            'GLOBAL_ARGUMENTS': GLOBAL_ARGUMENTS,
            'format_comma': format_comma,
            'format_duration': format_duration,
            'format_percent': format_percent,
            'format_row': format_row
        }

        f.write(template.render(**context).encode('utf-8'))