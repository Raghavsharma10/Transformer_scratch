def get_aggregate(self):
        """ Get aggregate data.

            Returns:
                A dict in with queries as keys and results as values.
        """
        return dict([(aggregate.find('query').text, [(ET.tostring(data).lstrip('<data xmlns:cps="www.clusterpoint.com" xmlns:cpse="www.clusterpoint.com">').strip().rstrip("</data>")) for data in aggregate.findall('data')])
                   for aggregate in self._content.findall('aggregate')])