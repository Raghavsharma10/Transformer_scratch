def get_facets(self):
        """ Get facets from the response.

            Returns:
                A dict where requested facet paths are keys and a list of coresponding terms are values.
        """
        return dict([(facet.attrib['path'], [term.text
                                             for term in facet.findall('term')])
                     for facet in self._content.findall('facet')])