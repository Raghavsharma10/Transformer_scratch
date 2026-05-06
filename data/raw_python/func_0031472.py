def get_tissue_specificities(cls, entry):
        """
        get list of :class:`pyuniprot.manager.models.TissueSpecificity` object from XML node entry

        :param entry: XML node entry
        :return: models.TissueSpecificity object
        """
        tissue_specificities = []

        query = "./comment[@type='tissue specificity']/text"

        for ts in entry.iterfind(query):
            tissue_specificities.append(models.TissueSpecificity(comment=ts.text))

        return tissue_specificities