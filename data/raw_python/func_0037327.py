def gene_forms(self):
        """
        :return: List of strings for all available gene forms
        :rtype: list[str]
        """
        q = self.session.query(distinct(models.ChemGeneIxnGeneForm.gene_form))
        return [x[0] for x in q.all()]