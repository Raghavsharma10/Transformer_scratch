def get_alternative_full_names(cls, entry):
        """
        get list of models.AlternativeFullName objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.AlternativeFullName` objects
        """
        names = []
        query = "./protein/alternativeName/fullName"
        for name in entry.iterfind(query):
            names.append(models.AlternativeFullName(name=name.text))

        return names