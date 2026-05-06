def vocabulary(self, levels=None):
        """
        :param levels: An optional argument of declaring a single or
            comma-delimited list of levels is available, as seen in the example
            as 1. An example of a comma-delimited list of levels is 1,2,5,9.
        :type levels: str or None

        http://www.wanikani.com/api/v1.2#vocabulary-list
        """

        url = WANIKANI_BASE.format(self.api_key, 'vocabulary')
        if levels:
            url += '/{0}'.format(levels)
        data = self.get(url)

        if 'general' in data['requested_information']:
            for item in data['requested_information']['general']:
                yield Vocabulary(item)
        else:
            for item in data['requested_information']:
                yield Vocabulary(item)