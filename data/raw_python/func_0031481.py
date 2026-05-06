def get_accessions(cls, entry):
        """
        get list of models.Accession from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.Accession` objects
        """
        return [models.Accession(accession=x.text) for x in entry.iterfind("./accession")]