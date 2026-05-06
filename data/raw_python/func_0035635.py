def _generatePlanetSearchDict(self):
        """ Generates a search dictionary for planets by taking all names and 'flattening' them to the most compact form
        (lowercase, no spaces and dashes)
        """

        planetNameDict = {}
        for planet in self.planets:

            name = planet.name
            altnames = planet.params['altnames']
            altnames.append(name)  # as we also want the default name to be searchable

            for altname in altnames:
                reducedname = compactString(altname)
                planetNameDict[reducedname] = planet

        return planetNameDict