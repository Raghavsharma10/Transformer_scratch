def get_gene_name(cls, entry):
        """
        get primary gene name from XML node entry

        :param entry: XML node entry
        :return: str
        """
        gene_name = entry.find("./gene/name[@type='primary']")

        return gene_name.text if gene_name is not None and gene_name.text.strip() else None