def get_other_gene_names(cls, entry):
        """
        get list of `models.OtherGeneName` objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.models.OtherGeneName` objects
        """
        alternative_gene_names = []

        for alternative_gene_name in entry.iterfind("./gene/name"):

            if alternative_gene_name.attrib['type'] != 'primary':

                alternative_gene_name_dict = {
                    'type_': alternative_gene_name.attrib['type'],
                    'name': alternative_gene_name.text
                }

                alternative_gene_names.append(models.OtherGeneName(**alternative_gene_name_dict))

        return alternative_gene_names