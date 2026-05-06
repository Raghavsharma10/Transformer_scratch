def get_chemical_diseases(self, direct_evidence=None, inference_gene_symbol=None, inference_score=None,
                              inference_score_operator=None, cas_rn=None, chemical_name=None,
                              chemical_id=None, chemical_definition=None, disease_definition=None,
                              disease_id=None, disease_name=None, limit=None, as_df=False):
        """Get chemical–disease associations with inference gene
        
        :param direct_evidence: direct evidence
        :param inference_gene_symbol: inference gene symbol
        :param inference_score: inference score
        :param inference_score_operator: inference score operator
        :param cas_rn:
        :param chemical_name: chemical name
        :param chemical_id:
        :param chemical_definition:
        :param disease_definition:
        :param disease_id:
        :param disease_name: disease name
        :param int limit: maximum number of results
        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :return: list of :class:`pyctd.manager.database.models.ChemicalDisease` objects

        .. seealso::
            
            :class:`pyctd.manager.models.ChemicalDisease`

            which is linked to:
            :class:`pyctd.manager.models.Disease`
            :class:`pyctd.manager.models.Chemical`
        """
        q = self.session.query(models.ChemicalDisease)

        if direct_evidence:
            q = q.filter(models.ChemicalDisease.direct_evidence.like(direct_evidence))

        if inference_gene_symbol:
            q = q.filter(models.ChemicalDisease.inference_gene_symbol.like(inference_gene_symbol))

        if inference_score:
            if inference_score_operator == ">":
                q = q.filter_by(models.ChemicalDisease.inference_score > inference_score)
            elif inference_score_operator == "<":
                q = q.filter_by(models.ChemicalDisease.inference_score > inference_score)

        q = self._join_chemical(q, cas_rn=cas_rn, chemical_id=chemical_id, chemical_name=chemical_name,
                                chemical_definition=chemical_definition)

        q = self._join_disease(q, disease_definition=disease_definition, disease_id=disease_id,
                               disease_name=disease_name)

        return self._limit_and_df(q, limit, as_df)