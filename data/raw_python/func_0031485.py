def get_organism_hosts(cls, entry):
        """
        get list of `models.OrganismHost` objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.OrganismHost` objects
        """

        query = "./organismHost/dbReference[@type='NCBI Taxonomy']"
        return [models.OrganismHost(taxid=x.get('id')) for x in entry.iterfind(query)]