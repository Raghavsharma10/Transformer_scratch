def _title(self):
        """
        Create a title summarizing the pathogens and samples.

        @return: A C{str} title.
        """
        return (
            'Overall, proteins from %d pathogen%s were found in %d sample%s.' %
            (len(self.pathogenNames),
             '' if len(self.pathogenNames) == 1 else 's',
             len(self.sampleNames),
             '' if len(self.sampleNames) == 1 else 's'))