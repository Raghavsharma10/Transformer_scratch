def get_chemical(self, chemical_name=None, chemical_id=None, cas_rn=None, drugbank_id=None, parent_id=None,
                     parent_tree_number=None, tree_number=None, synonym=None, limit=None, as_df=False):
        """Get chemical

        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param str chemical_name: chemical name
        :param str chemical_id: cehmical identifier 
        :param str cas_rn: CAS registry number
        :param str drugbank_id: DrugBank identifier 
        :param str parent_id: identifiers of the parent terms
        :param str parent_tree_number: identifiers of the parent nodes
        :param str tree_number: identifiers of the chemical's nodes
        :param str synonym: chemical synonym
        :param int limit: maximum number of results 
        :return: list of :class:`pyctd.manager.models.Chemical` objects
        

        .. seealso::

            :class:`pyctd.manager.models.Chemical`
        """
        q = self.session.query(models.Chemical)

        if chemical_name:
            q = q.filter(models.Chemical.chemical_name.like(chemical_name))

        if chemical_id:
            q = q.filter(models.Chemical.chemical_id == chemical_id)

        if cas_rn:
            q = q.filter(models.Chemical.cas_rn == cas_rn)

        if drugbank_id:
            q = q.join(models.ChemicalDrugbank).filter(models.ChemicalDrugbank.drugbank_id == drugbank_id)

        if parent_id:
            q = q.join(models.ChemicalParentid).filter(models.ChemicalParentid.parent_id == parent_id)

        if tree_number:
            q = q.join(models.ChemicalTreenumber) \
                .filter(models.ChemicalTreenumber.tree_number == tree_number)

        if parent_tree_number:
            q = q.join(models.ChemicalParenttreenumber) \
                .filter(models.ChemicalParenttreenumber.parent_tree_number == parent_tree_number)

        if synonym:
            q = q.join(models.ChemicalSynonym).filter(models.ChemicalSynonym.synonym.like(synonym))

        return self._limit_and_df(q, limit, as_df)