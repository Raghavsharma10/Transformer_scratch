def parse(self, s, term_join=None):
        """ Parses search term to

        Args:
            s (str): string with search term.
            or_join (callable): function to join 'OR' terms.

        Returns:
            dict: all of the terms grouped by marker. Key is a marker, value is a term.

        Example:
            >>> SearchTermParser().parse('table2 from 1978 to 1979 in california')
            {'to': 1979, 'about': 'table2', 'from': 1978, 'in': 'california'}
        """

        if not term_join:
            term_join = lambda x: '(' + ' OR '.join(x) + ')'

        toks = self.scan(s)

        # Examples: starting with this query:
        # diabetes from 2014 to 2016 source healthindicators.gov

        # Assume the first term is ABOUT, if it is not marked with a marker.
        if toks and toks[0] and (toks[0][0] == self.TERM or toks[0][0] == self.QUOTEDTERM):
            toks = [(self.MARKER, 'about')] + toks


        # The example query produces this list of tokens:
        #[(3, 'about'),
        # (0, 'diabetes'),
        # (3, 'from'),
        # (4, 2014),
        # (3, 'to'),
        # (4, 2016),
        # (3, 'source'),
        # (0, 'healthindicators.gov')]

        # Group the terms by their marker.

        bymarker = []
        for t in toks:
            if t[0] == self.MARKER:
                bymarker.append((t[1], []))
            else:
                bymarker[-1][1].append(t)


        # After grouping tokens by their markers
        # [('about', [(0, 'diabetes')]),
        # ('from', [(4, 2014)]),
        # ('to', [(4, 2016)]),
        # ('source', [(0, 'healthindicators.gov')])
        # ]

        # Convert some of the markers based on their contents. This just changes the marker type for keywords
        # we'll do more adjustments later.
        comps = []
        for t in bymarker:

            t = list(t)

            if t[0] == 'in' and len(t[1]) == 1 and isinstance(t[1][0][1], string_types) and self.stem(
                    t[1][0][1]) in self.geograins.keys():
                t[0] = 'by'

            # If the from term isn't an integer, then it is really a source.
            if t[0] == 'from' and len(t[1]) == 1 and t[1][0][0] != self.YEAR:
                t[0] = 'source'

            comps.append(t)

        # After conversions
        # [['about', [(0, 'diabetes')]],
        #  ['from', [(4, 2014)]],
        #  ['to', [(4, 2016)]],
        #  ['source', [(0, 'healthindicators.gov')]]]

        # Join all of the terms into single marker groups
        groups = {marker: [] for marker, _ in comps}

        for marker, terms in comps:
            groups[marker] += [term for marker, term in terms]

        # At this point, the groups dict is formed, but it will have a list
        # for each marker that has multiple terms.

        # Only a few of the markers should have more than one term, so move
        # extras to the about group

        for marker, group in groups.items():

            if marker == 'about':
                continue

            if len(group) > 1 and marker not in self.multiterms:
                groups[marker], extras = [group[0]], group[1:]

                if not 'about' in groups:
                    groups['about'] = extras
                else:
                    groups['about'] += extras

            if marker == 'by':
                groups['by'] = [ self.geograins.get(self.stem(e)) for e in group]

        for marker, terms in iteritems(groups):

            if len(terms) > 1:
                if marker in 'in':
                    groups[marker] = ' '.join(terms)
                else:
                    groups[marker] = term_join(terms)
            elif len(terms) == 1:
                groups[marker] = terms[0]
            else:
                pass

        # After grouping:
        # {'to': 2016,
        #  'about': 'diabetes',
        #  'from': 2014,
        #  'source': 'healthindicators.gov'}

        # If there were any markers with multiple terms, they would be cast in the or_join form.


        return groups