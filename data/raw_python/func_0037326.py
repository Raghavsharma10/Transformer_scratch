def get_chem_gene_interaction_actions(self, gene_name=None, gene_symbol=None, gene_id=None, limit=None,
                                          cas_rn=None, chemical_id=None, chemical_name=None, organism_id=None,
                                          interaction_sentence=None, chemical_definition=None,
                                          gene_form=None, interaction_action=None, as_df=False):
        """Get all interactions for chemicals on a gene or biological entity (linked to this gene).

        Chemicals can interact on different types of biological entities linked to a gene. A list of allowed
        entities linked to a gene can be retrieved via the attribute :attr:`~.gene_forms`.

        Interactions are classified by a combination of interaction ('affects', 'decreases', 'increases') 
        and actions ('activity', 'expression', ...  ). A complete list of all allowed
        interaction_actions can be retrieved via the attribute :attr:`~.interaction_actions`.

        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param str interaction_sentence: sentence describing the interactions 
        :param int organism_id: NCBI TaxTree identifier. Example: 9606 for Human.
        :param str chemical_name: chemical name
        :param str chemical_id: chemical identifier
        :param str cas_rn: CAS registry number
        :param str chemical_definition:
        :param str gene_symbol: HGNC gene symbol
        :param str gene_name: gene name
        :param int gene_id: NCBI Entrez Gene identifier
        :param str gene_form: gene form
        :param str interaction_action: combination of interaction and actions
        :param int limit: maximum number of results
        :rtype: list[models.ChemGeneIxn]


        .. seealso::

            :class:`pyctd.manager.models.ChemGeneIxn` 

            which is linked to:
            :class:`pyctd.manager.models.Chemical`
            :class:`pyctd.manager.models.Gene`
            :class:`pyctd.manager.models.ChemGeneIxnPubmed`

            Available interaction_actions and gene_forms
            :func:`pyctd.manager.database.Query.interaction_actions`
            :func:`pyctd.manager.database.Query.gene_forms`

        """
        q = self.session.query(models.ChemGeneIxn)

        if organism_id:
            q = q.filter(models.ChemGeneIxn.organism_id == organism_id)

        if interaction_sentence:
            q = q.filter(models.ChemGeneIxn.interaction == interaction_sentence)

        if gene_form:
            q = q.join(models.ChemGeneIxnGeneForm).filter(models.ChemGeneIxnGeneForm.gene_form == gene_form)

        if interaction_action:
            q = q.join(models.ChemGeneIxnInteractionAction) \
                .filter(models.ChemGeneIxnInteractionAction.interaction_action.like(interaction_action))

        q = self._join_gene(query=q, gene_name=gene_name, gene_symbol=gene_symbol, gene_id=gene_id)

        q = self._join_chemical(query=q, cas_rn=cas_rn, chemical_id=chemical_id, chemical_name=chemical_name,
                                chemical_definition=chemical_definition)

        return self._limit_and_df(q, limit, as_df)