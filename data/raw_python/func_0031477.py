def get_alternative_short_names(cls, entry):
        """
        get list of models.AlternativeShortName objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.AlternativeShortName` objects
        """
        names = []
        query = "./protein/alternativeName/shortName"
        for name in entry.iterfind(query):
            names.append(models.AlternativeShortName(name=name.text))

        return names