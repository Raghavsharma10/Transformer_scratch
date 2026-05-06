def get_gene_disease(self, direct_evidence=None, inference_chemical_name=None, inference_score=None,
                         gene_name=None, gene_symbol=None, gene_id=None, disease_name=None, disease_id=None,
                         disease_definition=None, limit=None, as_df=False):
        """Get gene–disease associations

        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param int gene_id: gene identifier
        :param str gene_symbol: gene symbol
        :param str gene_name:  gene name
        :param str direct_evidence: direct evidence
        :param str inference_chemical_name: inference_chemical_name 
        :param float inference_score: inference score
        :param str inference_chemical_name: chemical name
        :param disease_name: disease name
        :param disease_id: disease identifier 
        :param disease_definition: disease definition 
        :param int limit: maximum number of results
        :return: list of :class:`pyctd.manager.database.models.GeneDisease` objects

        .. seealso::
            
            :class:`pyctd.manager.models.GeneDisease`

            which is linked to:
            :class:`pyctd.manager.models.Chemical`
            :class:`pyctd.manager.models.Gene`
        """
        q = self.session.query(models.GeneDisease)

        if direct_evidence:
            q = q.filter(models.GeneDisease.direct_evidence == direct_evidence)

        if inference_chemical_name:
            q = q.filter(models.GeneDisease.inference_chemical_name == inference_chemical_name)

        if inference_score:
            q = q.filter(models.GeneDisease.inference_score == inference_score)

        q = self._join_disease(query=q, disease_definition=disease_definition, disease_id=disease_id,
                               disease_name=disease_name)

        q = self._join_gene(q, gene_name=gene_name, gene_symbol=gene_symbol, gene_id=gene_id)

        return self._limit_and_df(q, limit, as_df)