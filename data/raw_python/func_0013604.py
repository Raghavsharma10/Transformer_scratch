def setSpeciesFromJson(self, speciesJson):
        """
        Sets the species, an OntologyTerm, to the specified value, given as
        a JSON string.

        See the documentation for details of this field.
        """
        try:
            parsed = protocol.fromJson(speciesJson, protocol.OntologyTerm)
        except:
            raise exceptions.InvalidJsonException(speciesJson)
        self._species = protocol.toJsonDict(parsed)