def radicals(self, levels=None):
        """
        :param levels string: An optional argument of declaring a single or
            comma-delimited list of levels is available, as seen in the example
            as 1. An example of a comma-delimited list of levels is 1,2,5,9.

        http://www.wanikani.com/api/v1.2#radicals-list
        """
        url = WANIKANI_BASE.format(self.api_key, 'radicals')
        if levels:
            url += '/{0}'.format(levels)
        data = self.get(url)

        for item in data['requested_information']:
            yield Radical(item)