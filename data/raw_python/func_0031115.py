def legendLabel(self):
        """
        Provide a textual description of the feature and its qualifiers to be
        used as a label in a plot legend.

        @return: A C{str} description of the feature.
        """
        excludedQualifiers = set((
            'codon_start', 'db_xref', 'protein_id', 'region_name',
            'ribosomal_slippage', 'rpt_type', 'translation', 'transl_except',
            'transl_table')
        )
        maxValueLength = 30
        result = []
        if self.feature.qualifiers:
            for qualifier in sorted(self.feature.qualifiers.keys()):
                if qualifier not in excludedQualifiers:
                    value = ', '.join(self.feature.qualifiers[qualifier])
                    if qualifier == 'site_type' and value == 'other':
                        continue
                    if len(value) > maxValueLength:
                        value = value[:maxValueLength - 3] + '...'
                    result.append('%s: %s' % (qualifier, value))
        return '%d-%d %s%s.%s' % (
            int(self.feature.location.start),
            int(self.feature.location.end),
            self.feature.type,
            ' (subfeature)' if self.subfeature else '',
            ' ' + ', '.join(result) if result else '')