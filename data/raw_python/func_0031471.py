def get_tissue_in_references(self, entry):
        """
        get list of models.TissueInReference from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.TissueInReference` objects
        """
        tissue_in_references = []
        query = "./reference/source/tissue"
        tissues = {x.text for x in entry.iterfind(query)}

        for tissue in tissues:

            if tissue not in self.tissues:
                self.tissues[tissue] = models.TissueInReference(tissue=tissue)
            tissue_in_references.append(self.tissues[tissue])

        return tissue_in_references